import argparse
import os


parser = argparse.ArgumentParser("Tokenization Launcher.")

# parser.add_argument("data_path", type=str, help="Path to the data to tokenize.")
parser.add_argument("output_name", type=str, help="Output name.")
parser.add_argument("--n_tasks", type=int, help="nb of tokenization tasks", default=1)
parser.add_argument("--max_toks", type=int, help="max tokens per file", default=1e9)
parser.add_argument("--tokenizer", type=str, help="tokenizer to use", default="huggingfacetb/cosmo2-tokenizer")
parser.add_argument("--text_key", type=str, default="paragraph")


if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import SamplerFilter
    from datatrove.pipeline.readers import ParquetReader, JsonlReader
    from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
    from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
    
    dic = {
        "PubMed2":"/ai_studio/datasource/PubMed2_raw",
    }
    for name, path in dic.items():
        dist_executor = SlurmPipelineExecutor(
            job_name=f"tokeniz-{name}",
            pipeline=[
                ParquetReader(
                    path, # read directly from huggingface
                    glob_pattern="*.parquet", # "**/*.parquet", 
                    text_key=args.text_key,
                ),
                #SamplerFilter(rate=0.5),
                DocumentTokenizer(
                    output_folder=f"/ai_studio/datasource/{name}",
                    tokenizer_name_or_path=args.tokenizer,
                    batch_size=10000,
                    max_tokens_per_file=args.max_toks,
                    shuffle=True,
                ),
            ],
            tasks=args.n_tasks,
            time="6:00:00",
            partition="cpu_long",
            logging_dir=f"/ai_studio/logs/{name}",
            cpus_per_task=32,
            mem_per_cpu_gb=2,
            mail_user= <email>,
        )
        dist_executor.run()
