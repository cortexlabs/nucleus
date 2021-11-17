# Translator API

This project implements a multi-lingual translation API, supporting translations between over 150 languages, using +1,000 large pre-trained models served using Nucleus:


```bash
curl https://localhost:8888/translator -X POST -H "Content-Type: application/json" -d
{"source_language": "en", "destination_language": "phi", "text": "It is a mistake to think you can solve any major problems just with potatoes." }

{"generated_text": "Sayop an paghunahuna nga masulbad mo ang bisan ano nga dagkong mga problema nga may patatas lamang."}
```

Priorities of this project include:

- __Cost effectiveness.__ Each language-to-language translation is handled by a different ~300 MB model. Traditional setups would deploy all +1,000 models across many servers to ensure availability, but this API can be run on a single server thanks to Nucleus' multi-model caching.
- __Ease of use.__ Predictions are generated using Hugging Face's Transformer Library and Nucleus' Handler API.
- __Configurability.__ All tools used in this API are fully open source and modifiable. The prediction API can be run on CPU and GPU instances.

## Models used

This project uses pre-trained Opus MT neural machine translation models, trained by Jörg Tiedemann and the Language Technology Research Group at the University of Helsinki. The models are hosted for free by Hugging Face. For the full list of language-to-language models, you can view the model repository [here.](https://huggingface.co/Helsinki-NLP)

Once the model server has indexed all +1,000 models, we can now query the API at the endpoint given, structuring the body of our request according to the format expected by our handler (specified in `handler.py`):

```
{
    "source_language": "en",
    "destination_language": "es",
    "text": "So long and thanks for all the fish."
}
```

The response should look something like this:

```
{"generated_text": "Hasta luego y gracias por todos los peces."}
```

The API, as currently defined, uses the two-letter codes used by the Helsinki NLP team to abbreviate languages. If you're unsure of a particular language's code, check the model names. Additionally, you can easily implement logic on the frontend or within your API itself to parse different abbreviations.

## Performance

The first time you request a specific language-to-language translation, the model will be downloaded from S3, which may take some time (~60s, depending on bandwidth). Every subsequent request can be much faster, as the API can be defined to hold 250 models on disk and 5 in memory. Models already loaded into memory will serve predictions fastest (a couple seconds at most with GPU), while those on disk will take slightly longer as they need to be swapped into memory. Instances with more memory and disk space can naturally hold more models.

As for caching logic, when space is full, models are removed from both memory and disk according to which model was used last.

Finally, note that this project places a heavy emphasis on cost savings, to the detriment of optimal performance. If you are interested in improving performance, there are a number of changes you can make. For example, if you know which models are most likely to be needed, you can "warm up" the API by calling them immediately after deploy. Alternatively, if you have a handful of translation requests that comprise the bulk of your workload, you can deploy a separate API containing just those models, and route traffic accordingly. You will increase cost (though still benefit greatly from multi-model caching), but you will also significantly improve the overall latency of your system.
