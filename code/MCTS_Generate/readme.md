## Quick Start

### Prepare Data

The source sentence should be formatted according to the following structure:

**en-zh**

```
But the victim's brother says he can't think of anyone who would want to hurt him, saying, "Things were finally going well for him."
The San Francisco Police Department said the death was ruled a homicide and an investigation is ongoing.
The victim's brother, Louis Galicia, told ABC station KGO in San Francisco that Frank, previously a line cook in Boston, had landed his dream job as line chef at San Francisco's Sons & Daughters restaurant six months ago.
A spokesperson for Sons & Daughters said they were "shocked and devastated" by his death.
...
```

### Prefixed Preference Pair Generation

Following data formatting, the program can be executed through two methods: **command-line interface** and **Python scripting**.

#### Command-line Interface

**Example**:

```
python MCTS.py src/zh-en output/result en zh
```

<hr>

```
usage: python MCTS.py [input_file] [output_file] [src_lang] [tar_lang]
		      [--gpu_map GPU_IDX] [--trans_model TRANS_MODEL] [--eva_model EVA_MODEL] [--threshold MIN MAX]
```

##### Named Arguments

`input_file` (required)

    The path of the translation source sentences.

`output_file` (required)

    The path of the output file.

`src_lang` (required)

    The language of the source sentences.
    choose from ['en', 'de', 'ru', 'zh']

    If you need to support additional languages, you can adjust the code:
```python
...
lang_name = {'en':'English', 'de':'German', 'ru':'Russian', 'zh':'Chinese'}  # e.g. 'ja':'Japanese'
...
```


