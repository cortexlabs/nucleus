# Multi-Model Classifier API

This example deploys Iris, ResNet50 and Inception models in one API. Query parameters are used for selecting the model.

Since model caching is enabled, there can only be 1 model loaded into memory - loading a 2rd one will lead to the removal of the least recently used one.

The example can be run on both CPU and on GPU hardware.

## Sample Prediction

When making a prediction with [sample-image.json](sample-image.json), the following image will be used:

![sports car](https://i.imgur.com/zovGIKD.png)

### ResNet50 Classifier

Make a request to the ResNet50 model:

```bash
curl "http://localhost:8080/?model=resnet50" -X POST -H "Content-Type: application/json" -d @sample-image.json
```

The expected response is:

```json
{"label": "sports_car"}
```

### Inception Classifier

Loading the following model will evict the previously loaded model from memory because the `cache_size` is set to 1. No model from disk is removed.

Make a request to the Inception model:

```bash
curl "http://localhost:8080/?model=inception" -X POST -H "Content-Type: application/json" -d @sample-image.json
```

The expected response is:

```json
{"label": "sports_car"}
```

### Iris Classifier

At this point, there are 2 models loaded onto disk (as specified by `disk_cache_size`). Loading the `iris` classifier will lead to the removal of the least recently used model - in this case, it will be the ResNet50 model that will get evicted.

Make a request to the Iris model:

```bash
curl "${ENDPOINT}?model=iris" -X POST -H "Content-Type: application/json" -d @sample-iris.json
```

The expected response is:

```json
{"label": "setosa"}
```
