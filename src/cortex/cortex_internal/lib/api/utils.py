# Copyright 2021 Cortex Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import glob
import itertools
import json
import os
import shutil
import threading as td
import time
import traceback
from collections import defaultdict
from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional, Tuple

from datadog.dogstatsd.base import DogStatsd
from starlette.responses import Response

from cortex_internal.lib import util
from cortex_internal.lib.exceptions import CortexException, UserRuntimeException
from cortex_internal.lib.log import configure_logger
from cortex_internal.lib.model import validate_model_paths
from cortex_internal.lib.storage import S3
from cortex_internal.lib.type import HandlerType

logger = configure_logger("cortex", os.environ["CORTEX_LOG_CONFIG_FILE"])


def model_downloader(
    handler_type: HandlerType,
    bucket_name: str,
    model_name: str,
    model_version: str,
    model_path: str,
    temp_dir: str,
    model_dir: str,
) -> Optional[datetime.datetime]:
    """
    Downloads model to disk. Validates the s3 model path and the downloaded model.

    Args:
        handler_type: The handler type as implemented by the API.
        bucket_name: Name of the bucket where the model is stored.
        model_name: Name of the model. Is part of the model's local path.
        model_version: Version of the model. Is part of the model's local path.
        model_path: Model prefix of the versioned model.
        temp_dir: Where to temporarily store the model for validation.
        model_dir: The top directory of where all models are stored locally.

    Returns:
        The model's timestamp. None if the model didn't pass the validation, if it doesn't exist or if there are not enough permissions.
    """

    logger.info(
        f"downloading from bucket {bucket_name}/{model_path}, model {model_name} of version {model_version}, temporarily to {temp_dir} and then finally to {model_dir}"
    )

    client = S3(bucket_name)

    # validate upstream S3 model
    sub_paths, ts = client.search(model_path)
    try:
        validate_model_paths(sub_paths, handler_type, model_path)
    except CortexException:
        logger.info(f"failed validating model {model_name} of version {model_version}")
        return None

    # download model to temp dir
    temp_dest = os.path.join(temp_dir, model_name, model_version)
    try:
        client.download_dir_contents(model_path, temp_dest)
    except CortexException:
        logger.info(
            f"failed downloading model {model_name} of version {model_version} to temp dir {temp_dest}"
        )
        shutil.rmtree(temp_dest)
        return None

    # validate model
    model_contents = glob.glob(os.path.join(temp_dest, "**"), recursive=True)
    model_contents = util.remove_non_empty_directory_paths(model_contents)
    try:
        validate_model_paths(model_contents, handler_type, temp_dest)
    except CortexException:
        logger.info(
            f"failed validating model {model_name} of version {model_version} from temp dir"
        )
        shutil.rmtree(temp_dest)
        return None

    # move model to dest dir
    model_top_dir = os.path.join(model_dir, model_name)
    ondisk_model_version = os.path.join(model_top_dir, model_version)
    logger.info(
        f"moving model {model_name} of version {model_version} to final dir {ondisk_model_version}"
    )
    if os.path.isdir(ondisk_model_version):
        shutil.rmtree(ondisk_model_version)
    shutil.move(temp_dest, ondisk_model_version)

    return max(ts)


class DynamicBatcher:
    def __init__(
        self,
        handler_impl: Callable,
        method_name: str,
        max_batch_size: int,
        batch_interval: int,
        test_mode: bool = False,
    ):
        self.method_name = method_name
        self.handler_impl = handler_impl

        self.batch_max_size = max_batch_size
        self.batch_interval = batch_interval  # measured in seconds
        self.test_mode = test_mode  # only for unit testing
        self._test_batch_lengths = []  # only when unit testing

        self.barrier = td.Barrier(self.batch_max_size + 1)

        self.samples = {}
        self.results = {}
        td.Thread(target=self._batch_engine, daemon=True).start()

        self.sample_id_generator = itertools.count()

    def _batch_engine(self):
        while True:
            if len(self.results) > 0:
                time.sleep(0.001)
                continue

            try:
                self.barrier.wait(self.batch_interval)
            except td.BrokenBarrierError:
                pass

            self.results = {}
            sample_ids = self._get_sample_ids(self.batch_max_size)
            try:
                if self.samples:
                    batch = self._make_batch(sample_ids)

                    results = getattr(self.handler_impl, self.method_name)(**batch)
                    if not isinstance(results, list):
                        raise UserRuntimeException(
                            f"please return a list when using server side batching, got {type(results)}"
                        )

                    if self.test_mode:
                        self._test_batch_lengths.append(len(results))

                    self.results = dict(zip(sample_ids, results))
            except Exception as e:
                self.results = {sample_id: e for sample_id in sample_ids}
                logger.error(traceback.format_exc())
            finally:
                for sample_id in sample_ids:
                    del self.samples[sample_id]
                self.barrier.reset()

    def _get_sample_ids(self, max_number: int) -> List[int]:
        if len(self.samples) <= max_number:
            return list(self.samples.keys())
        return sorted(self.samples)[:max_number]

    def _make_batch(self, sample_ids: List[int]) -> Dict[str, List[Any]]:
        batched_samples = defaultdict(list)
        for sample_id in sample_ids:
            for key, sample in self.samples[sample_id].items():
                batched_samples[key].append(sample)

        return dict(batched_samples)

    def _enqueue_request(self, sample_id: int, **kwargs):
        """
        Enqueue sample for batch processing. This is a blocking method.
        """

        self.samples[sample_id] = kwargs
        try:
            self.barrier.wait()
        except td.BrokenBarrierError:
            pass

    def process(self, **kwargs):
        """
        Queues a request to be batched with other incoming request, waits for the response
        and returns the processed result. This is a blocking method.
        """
        sample_id = next(self.sample_id_generator)
        self._enqueue_request(sample_id, **kwargs)
        result = self._get_result(sample_id)
        return result

    def _get_result(self, sample_id: int) -> Any:
        """
        Return the processed result. This is a blocking method.
        """
        while sample_id not in self.results:
            time.sleep(0.001)

        result = self.results[sample_id]
        del self.results[sample_id]

        if isinstance(result, Exception):
            return Response(
                content=str(result),
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                media_type="text/plain",
            )

        return result
