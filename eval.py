import argparse
import collections
import itertools
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default='datasets/synth_test.jsonl')
    parser.add_argument('--model', type=Path, default='out/LLM-JEPA-135M')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--out', type=Path, default='eval')

    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    with open(path, 'r') as file:
        return [json.loads(line) for line in file]


def batched(iterable, n):
    iterator = iter(iterable)

    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


def save_responses(responses: list[dict], out: Path) -> None:
    out = out / 'responses.jsonl'

    with open(out, 'w') as file:
        for response in responses:
            file.write(json.dumps(response, ensure_ascii=False) + '\n')

    print(f'Saved responses to {out}')


def save_metrics(args: argparse.Namespace, accuracy: float, out: Path) -> None:
    out = out / 'metrics.json'

    metrics = {
        'model': args.model.as_posix(),
        'dataset': args.data.as_posix(),
        'accuracy': round(accuracy, 2),
    }

    with open(out, 'w') as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    print(f'Saved metrics to {out}')


@torch.inference_mode()
def evaluate():
    args = parse_arguments()

    dataset = load_dataset(args.data)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(dtype=torch.bfloat16, device='cuda')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left'  # Need left padding for generation

    responses = []

    progress = tqdm(total=len(dataset), desc='Evaluating')

    for batch in batched(dataset, args.batch_size):
        prompts = [sample['messages'][:-1] for sample in batch]
        prompts = tokenizer.apply_chat_template(
            conversation=prompts,
            padding='longest',
            return_tensors='pt',
            add_generation_prompt=True
        ).to(model.device)

        outputs = model.generate(
            prompts.to(device=model.device),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_new_tokens=args.max_tokens,
        )

        outputs = outputs[:, prompts.shape[1]:]

        generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = [sample['messages'][-1]['content'] for sample in batch]

        for generation, reference in zip(generations, references):
            responses.append({'generated': generation, 'reference': reference})

        progress.update(args.batch_size)

    counts = collections.defaultdict(int)

    for response in responses:
        generated = response['generated']
        reference = response['reference']

        response['correct'] = generated.strip() == reference.strip()

        if response['correct']:
            counts['correct'] += 1

        counts['total'] += 1

    accuracy = 100 * counts['correct'] / counts['total']
    print(f'Accuracy: {accuracy:.2f}% ({counts["correct"]}/{counts["total"]})')

    args.out.mkdir(exist_ok=True, parents=True)

    save_responses(responses, args.out)
    save_metrics(args, accuracy, args.out)


if __name__ == '__main__':
    evaluate()
