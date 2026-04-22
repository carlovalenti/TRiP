# TRiP — TRansformer in Progress

A few-files, all-in-one C engine for Transformer AI models: inference, training, tokenizer creation, chat, and vision.

Built from scratch over 18 months during my lunch breaks and weekend nights, TRiP exists because I just wanted to truly understand the transformer internals - from the matrix multiplications up.
TRiP's purpose is purely educational — for me and for anyone willing to learn about transformers. It supports Gemma 1, Llama 2, PaliGemma, and GPT-2, with full inference and training. It does not aim to track the latest model releases, and is not trying to compete with llama.cpp.

## What TRiP supports

- **Architectures:** Llama2, Gemma 1.0/1.1, PaliGemma 1 (vision+language), GPT-2
- **Checkpoint formats:** SafeTensors (HuggingFace), Karpathy's llama2.c and gpt2 formats
- **Weight types:** bf16, float16, float32
- **Training:** full backpropagation with AdamW, cosine annealing LR, gradient clipping
- **Tokenizer:** BPE (SentencePiece-compatible), with vocabulary creation from scratch
- **Inference:** greedy, top-k, and nucleus (top-p) sampling
- **Chat:** interactive chat with Llama, Gemma, and TinyLlama chat templates
- **Vision:** multimodal inference with PaliGemma (JPEG input, X11 display)
- **Memory:** RAM-optimized mode via mmap for large models on limited hardware

## Building

### Dependencies

```
gcc (recommended: version 13 or higher, to get support for bfloat16; with OpenMP support)
libjpeg-dev (or libjpeg62-turbo-dev)
libx11-dev
```

WARNING: do NOT expect higher performance with bfloat16 or float16 on CPUs; today's CPUs are not optimized for floating point operations in such formats, and float32 always performs best.
That surprised me a lot, too.


On Debian/Ubuntu:
```bash
sudo apt install build-essential libomp-dev libjpeg62-turbo-dev libx11-dev
```

### Windows (WSL)

TRiP runs natively under WSL (Windows Subsystem for Linux). To enable
the X11 display features (vision mode, image display), install an X server
on the Windows side:

- [VcXsrv](https://sourceforge.net/projects/vcxsrv/) (free, lightweight)
- [Xming](http://www.straightrunning.com/XmingNotes/) (free version available)
- [MobaXterm](https://mobaxterm.mobatek.net/) (has a built-in X server)

Then in your WSL terminal, before running TRiP:

```bash
export DISPLAY=:0
```

If using WSL2 (most setups), use instead:

```bash
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
```

X11 is only needed for vision mode. Chat, inference, and training work without it.


### Compile

```bash
make
```

That's it. No cmake, no external frameworks, no Python. Just `make`.

## Quick start

### Chat with a Gemma model

Download a Gemma-2B-IT model from HuggingFace (safetensors format), then:

```bash
./trip --chat \
    --checkpoint gemma-2b-it/model.safetensors \
    --tokenizer gemma-2b-it/tokenizer.json \
    --chat_scheme GEMMA
```

### Run inference on a prompt

```bash
./trip --decode \
    --input_text "The capital of Italy is" \
    --checkpoint gemma-2b-it/model.safetensors \
    --tokenizer gemma-2b-it/tokenizer.json
```

Or from a text file:

```bash
./trip --decode prompt.txt \
    --checkpoint gemma-2b-it/model.safetensors \
    --tokenizer gemma-2b-it/tokenizer.json
```

### Train a model

```bash
./trip --train \
    --checkpoint my_model/model.safetensors \
    --tokenizer my_model/tokenizer.json \
    --train_data my_dataset.txt \
    --train_config training_args.json
```

### Vision (PaliGemma)

```bash
./trip --vision photo.jpg \
    --checkpoint paligemma/model.safetensors \
    --tokenizer paligemma/tokenizer.json \
    --input_text "Describe this image"
```

### Build a tokenizer vocabulary from scratch

```bash
./trip --build_vocab corpus.txt --vocab_size 32000 --tokenizer my_tokenizer.json
```

## Full CLI reference

```
USAGE:
  ./trip <ACTION> [OPTIONS...]
```

### Actions (pick one)

| Flag | Description |
|------|-------------|
| `--decode [file]` | Run inference on a prompt (from file, `--input_text`, or stdin) |
| `--chat` | Interactive chat session |
| `--vision [image.jpg]` | Multimodal inference with an image |
| `--train` | Train the model |
| `--create` | Create a new model from a configuration file |
| `--build_vocab <data.txt>` | Build a new tokenizer vocabulary from a text corpus |
| `--utest` | Run unit tests |
| `--help` | Show help |

### Model & tokenizer options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint <path>` | `default.model` | Path to model checkpoint file(s) |
| `--checkpoint_type <type>` | `SAFETENSORS` | Format: `SAFETENSORS`, `LLAMA2_AK`, `GPT2_AK` |
| `--configuration <path>` | (auto) | Path to `config.json` (for SafeTensors) |
| `--tokenizer <path>` | `default.tokenizer` | Path to tokenizer file |
| `--tokenizer_format <type>` | `JSON_HUGGINGFACE` | Format: `JSON_HUGGINGFACE`, `LLAMA2_AK`, `GPT2_AK` |
| `--tokenizer_type <type>` | `SENTENCEPIECE` | Algorithm: `SENTENCEPIECE`, `TRIP` |

### Inference & sampling options

| Flag | Default | Description |
|------|---------|-------------|
| `--input_text "<prompt>"` | — | Provide prompt text directly on the command line |
| `--system_prompt "<text>"` | — | System prompt for chat mode |
| `--chat_scheme <scheme>` | (none) | Chat template: `LLAMA`, `TINY_LLAMA`, `GEMMA` |
| `--chat_save_context <file>` | — | Pre-process and save chat context for faster startup |
| `--chat_load_context <file>` | — | Load a previously saved chat context |
| `--temperature <value>` | `1.0` | Sampling temperature. `0.0` = greedy (always pick the most probable token) |
| `--top_p <value>` | `0.9` | Nucleus sampling: sample from the smallest set of tokens whose cumulative probability exceeds this value |
| `--top_k <value>` | (disabled) | Top-k sampling: sample from the k most probable tokens |
| `--ram` | (off) | Memory-map weights instead of loading them (slower, uses less RAM) |

### Training options

| Flag | Default | Description |
|------|---------|-------------|
| `--train_config <path>` | `training_args.json` | Path to training configuration JSON |
| `--train_data <path>` | `training_data.txt` | Path to training data (plain text) |

## File map

TRiP is organized into 7 files. Open `trip.h` for the complete map.

| File | Lines | What it contains |
|------|-------|-----------------|
| `trip.h` | ~900 | The map. Every type, struct, global, and declaration. |
| `math.c` | ~3000 | Tensor ops, each forward+backward paired side by side: matmul, softmax, layernorm, RMSnorm, RoPE, attention, FFN activations, vector arithmetic |
| `forward.c` | ~1500 | Forward pass orchestration + token sampling |
| `backward.c` | ~1500 | Backward pass + AdamW optimizer + gradient management |
| `model.c` | ~5500 | Checkpoint I/O, model init, memory management, tokenizer, vision preprocessing |
| `utils.c` | ~1000 | Logging, JSON parser, terminal I/O, JPEG/X11 image handling |
| `main.c` | ~1900 | CLI argument parsing, chat loop, training loop, inference loop |

## How it works (for the curious)

TRiP implements a transformer from first principles in C. No PyTorch, no TensorFlow, no ONNX — just linear algebra on arrays of floats.

The **residual stream** is the central concept: a vector that flows through the model like data on a bus. Each layer reads from it, processes it through attention and a feed-forward network, and writes back to it. The forward pass walks the layers top to bottom; the backward pass walks them bottom to top, computing gradients via the chain rule.

Every math operation (`math.c`) is implemented as a forward+backward pair: you can read `rmsnorm()` and immediately below it `rmsnorm_backward()`, and see exactly how the gradient flows through the same computation in reverse.

I put a lot of comments in the code, both as reminders to me, and to render TRiP basically an annotated school book about transformers.

For a deeper understanding of backpropagation, see [Andrej Karpathy's lecture](https://www.youtube.com/watch?v=i94OvYb6noo); TRiP would never have existed without his work.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free to use, study, modify, and share for non-commercial purposes, with attribution. For commercial licensing, contact the author.

## Acknowledgments

- **Andrej Karpathy** — for [llama2.c](https://github.com/karpathy/llama2.c), [nanoGPT](https://github.com/karpathy/nanoGPT), and the lectures that made all of this possible
- **Google** — for releasing the Gemma model family
- **Meta** — for releasing the Llama model family

