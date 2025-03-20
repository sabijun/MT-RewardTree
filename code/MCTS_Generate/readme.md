# Quick Start

## Prepare Data

The source sentence should be formatted according to the following structure:

**en-zh**

```
But the victim's brother says he can't think of anyone who would want to hurt him, saying, "Things were finally going well for him."
The San Francisco Police Department said the death was ruled a homicide and an investigation is ongoing.
The victim's brother, Louis Galicia, told ABC station KGO in San Francisco that Frank, previously a line cook in Boston, had landed his dream job as line chef at San Francisco's Sons & Daughters restaurant six months ago.
A spokesperson for Sons & Daughters said they were "shocked and devastated" by his death.
...
```

## Prefixed Preference Pair Generation

Following data formatting, the program can be executed through two methods: **command-line interface** and **Python scripting**.

### Command-line Interface

**Example**:

```
python MCTS.py src/en-zh output/result en zh --gpu_map 0,1 --threshold 0.04 0.4
```

<hr>

```
usage: python MCTS.py [input_file] [output_file] [src_lang] [tar_lang]
		      [--gpu_map GPU_IDX] [--trans_model TRANS_MODEL] [--eva_model EVA_MODEL] [--threshold MIN MAX]
```

##### Named Arguments

`input_file` (required)
    
​	The path of the translation source sentences.

`output_file` (required)

​	The path of the output file.

`src_lang` (required)

​	The language of the source sentences.

​	Possible choices: "en", "de", "ru", "zh"

​	If you need to support additional languages, you can adjust the code:
```python
...
lang_name = {'en':'English', 'de':'German', 'ru':'Russian', 'zh':'Chinese'}  # e.g. 'jp':'Japanese'
...
```
`tar_lang` (required)

​	The language of the target sentences.

​	Same as `src_lang`.

`--gpu_map` (optional)

​	The gpu index used.

​	Default: "0,1"

`--trans_model` (optional)

​	The name/path of the translation model.

​	Default: "Unbabel/TowerInstruct-7B-v0.2"

> [!NOTE]
> If you need to modify the prompt, you can access the `process` method.

`--eva_model` (optional)

​	the name/path of the evaluation model.

​	Default: "Unbabel/wmt22-cometkiwi-da"

> [!NOTE]
> We recommend using cometkiwi series.
> If you wish to use any alternative reference-free evaluation, you may need to modify the `get_score` method accordingly.

`--threshold` (optional)

​	The score threshold that determine if the translation pair will be logged.

​	Default: 0.04 0.4

### python code usage

You can find the example in [usage.ipynb](usage.ipynb)


