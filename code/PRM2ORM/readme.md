# Quick Start

## Prepare Data

The source sentence should be formatted according to the following structure:

**en-zh**

```
The hacked up version of Jedi Knight was crashing because it was calling a function off the end of a vtable.
绝地武士的破解版崩溃了，因为它调用了 vtable 结尾的一个函数。
Even if it's true that such facts exist in science it's still possible to argue that scientific facts are theory-laden.
即使这些事实确实存在于科学之中，仍然有可能认为科学事实是纯理论的。
...
```

## Process RM to Sequence-level Scoring

Following data formatting, the program can be executed through two methods: **command-line interface** and **Python scripting**.

### Command-line Interface

**Example**:

```
python PRM2ORM.py sabijun/MT-PRM-LLaMA-3.2-3B meta-llama/Llama-3.2-3B-Instruct src/en-zh en zh --trained_device 0 --ref_device 1 --method weighted
```

<hr>


```
usage: python PRM2ORM.py [trained_model] [ref_model] [input_path] [src_lang] [tar_lang]
                         [--trained_device TRAINED_GPU_IDX] [--ref_device REF_GPU_IDX] [--method METHOD]
```

##### Named Arguments

`trained_model` (required)

    The name/path of the trained model.

​	Possible choices: `sabijun/MT-PRM-LLaMA-3.2-3B` , `sabijun/MT-PRM-Qwen-2.5-3B`

`ref_model` (required)

​	The name/path of the reference model.

​	Possible choices: `meta-llama/Llama-3.2-3B-Instruct` , `Qwen/Qwen2.5-3B-Instruct`

> [!NOTE]
> The trained model must share the same vocabulary as the reference model.
>
> You can use our models, which are trained based on **Llama-3.2-3B-Instruct** and **Qwen-2.5-3B-Instruct**

`input_path` (required)

​	The path of the source and hypothesis.

`src_lang` (required)

​	The language of the source sentences.

​	Possible choices: `en` , `de` , `ru` , `zh`

​	If you need to support additional languages, you can adjust the code:

```python
...
lang_name = {'en':'English', 'de':'German', 'ru':'Russian', 'zh':'Chinese'}  # e.g. 'jp':'Japanese'
...
```

`tar_lang` (required)

​	The language of the target sentences.

​	Same as `src_lang`.

`--trained_device` (optional)

​	The gpu index used.

​	Default: "0"

`--ref_device` (optional)

​	The gpu index used.

​	Default: "1"

`--method` (optional)

​	The method of Sequence-level score calculation.

​	Possible choices: `weighted` , `length normalized`

​	Default: "weighted"

### python code usage

You can find the example in [usage.ipynb](usage.ipynb)