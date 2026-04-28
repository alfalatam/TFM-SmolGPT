import argparse
from smolgpt.ask import main as ask_main
from smolgpt.check_vocab import main as check_vocab_main
from smolgpt.check_token_number import main as check_token_number_main
from smolgpt.count_bos_eos import main as count_bos_eos_main
from smolgpt.tokenization_example import main as tokenization_example_main
from smolgpt.export_vocabulary import main as export_vocabulary_main
from smolgpt.script_tokenizer import main as script_tokenizer_main
#from smolgpt.evaluate import main as evaluate_main
from smolgpt.preprocess import train_vocab_txt
from smolgpt.test import main as test_main

def ask():
    ask_main()

def check_vocab():
    check_vocab_main()

def check_token_number():
    check_token_number_main()

def count_bos_eos():
    count_bos_eos_main()

def tokenization_example():
    tokenization_example_main()


def export_vocabulary():
    export_vocabulary_main()

def script_tokenizer():
    script_tokenizer_main()
    
# def evaluate():
    # evaluate_main()
def test():
    test_main()


# =====================
def main():
    parser = argparse.ArgumentParser(
        description="Utility tools for SmolGPT"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    train_vocab_txt_parser = subparsers.add_parser(
    "train_vocab_txt",
    help="Train a tokenizer vocabulary from .txt files"
    )
    
    train_vocab_txt_parser.add_argument(
        "--vocab-size",
        type=int, 
        required=True, 
        help="vocabulary size"
    )

    train_vocab_txt_parser.add_argument(
        "--txt-dir",
        type=str,
        required=True,
        help="where the .txt files are stored"
    )
        
    subparsers.add_parser(
        "ask",
        help="Enter interactive Q&A mode to query the model"
    )

    subparsers.add_parser(
        "check_vocab",
        help="Check if tokens are present in the tokenizer vocabulary"
    )

    subparsers.add_parser(
        "check_token_number",
        help="Check the number of generated tokens and the unique ones"
    )

    subparsers.add_parser(
        "count_bos_eos",
        help="Count BOS and EOS token occurrences"
    )

    subparsers.add_parser(
        "tokenization_example",
        help="Let the user tokenize a word or setence to test it"
    )

    subparsers.add_parser(
        "export_vocabulary",
        help="Export the vocabulary to a text file"
    )

    subparsers.add_parser(
        "script_tokenizer",
        help="Run the tokenizer script for the text file in the data folder is recommended to have only one but you can concatenate multiple ones"
    )
    subparsers.add_parser(
        "test",
        help="Run a multiple choice test"
    )   

    # subparsers.add_parser(
        # "evaluate",
        # help="Runs a small test type to check the model in inference"
    # )

    args = parser.parse_args()

    if args.command == "ask":
        ask()
    elif args.command == "check_vocab":
        check_vocab()
    elif args.command == "check_token_number":
        check_token_number()
    elif args.command == "count_bos_eos":
        count_bos_eos()
    # elif args.command == "evaluate":
        # evaluate()
    elif args.command == "train_vocab_txt":
        train_vocab_txt(args.vocab_size, args.txt_dir)
    elif args.command == "tokenization_example":
        tokenization_example()
    elif args.command == "export_vocabulary":
        export_vocabulary()
    elif args.command == "script_tokenizer":
        script_tokenizer()
    elif args.command == "test":
        test()

 
if __name__ == "__main__":
    main()  