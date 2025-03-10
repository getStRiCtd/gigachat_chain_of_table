# Copyright 2024 The Chain-of-Table authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import fire
import asyncio

from src.chain_of_table.utils.load_data import load_tabfact_dataset
from src.chain_of_table.utils.llm import GigaChat_CoT
from src.chain_of_table.utils.evaluate import *
from src.chain_of_table.utils.chain import *
from src.chain_of_table.operations import *


def main(
    dataset_path: str = "../data/tabfact/test.jsonl",
    raw2clean_path: str ="../data/tabfact/raw2clean.jsonl",
    result_dir: str = "../results/tabfact",
    first_n=-1,
    n_proc=1,
    chunk_size=1,
):
    dataset = load_tabfact_dataset(dataset_path, raw2clean_path, first_n=first_n)
    giga_cot = GigaChat_CoT()
    os.makedirs(result_dir, exist_ok=True)

    proc_samples, dynamic_chain_log_list = dynamic_chain_exec_with_cache_mp(
        dataset,
        llm=giga_cot,
        llm_options=giga_cot.get_model_options(
            temperature=0.0, per_example_max_decode_steps=200, per_example_top_p=1.0
        ),
        strategy="top",
        cache_dir=os.path.join(result_dir, "cache"),
        n_proc=n_proc,
        chunk_size=chunk_size,
    )
    fixed_chain = [
        (
            "simpleQuery_fewshot",
            simple_query,
            dict(use_demo=True),
            dict(
                temperature=0, per_example_max_decode_steps=200, per_example_top_p=1.0
            ),
        ),
    ]
    final_result, _ = fixed_chain_exec_mp(giga_cot, proc_samples, fixed_chain)
    acc = tabfact_match_func_for_samples(final_result)
    print("Accuracy:", acc)

    print(
        f'Accuracy: {acc}',
        file=open(os.path.join(result_dir, "result.txt"), "w")
    )
    pickle.dump(
        final_result, open(os.path.join(result_dir, "final_result.pkl"), "wb")
    )
    pickle.dump(
        dynamic_chain_log_list, 
        open(os.path.join(result_dir, "dynamic_chain_log_list.pkl"), "wb")
    )


if __name__ == "__main__":
    import pickle
    with open("/media/ntfs_disk/Pycharn/test_vllm/results/tabfact/final_result.pkl", 'rb') as f:
        d = pickle.load(f)
        print(d)
    # fire.Fire(main)
