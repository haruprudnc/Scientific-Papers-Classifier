import argparse

import numpy as np
import pandas as pd

from config import config
from utils.cli_progress_bar import cliProgressBar
from utils.color import Color
from utils.terminal import get_terminal_length

from core.crawler import crawl_data, precrawl_process
from core.preprocessor import preprocess_data
from core.trainer import train_model, train_model_specter_classifier
from core.predictor import predict








# .-. ======================================== MAIN PROGRAM ======================================== .-. #
if __name__ == "__main__":
    startup_msg = "    world.execute(me)    ".center(get_terminal_length(), "=")
    print(f"{Color.pink}{Color.bold}{startup_msg}{Color.reset}")

    # ^^ ======================================== PARSER ======================================== ^^ #
    parser = argparse.ArgumentParser(description="A simple argument parser.")

    parser.add_argument("-pc", "--precrawl", action = "store_true", help = "Default is False, use to start precrawl process data.")
    parser.add_argument("-c", "--crawl", action = "store_true", help = "Default is False, use to start crawling data.")
    parser.add_argument("-p", "--preprocess", action = "store_true", help = "Default is False, use to start preprocessing data.")
    parser.add_argument("-t", "--train", action = "store_true", help = "Default is False, use to start training model.")
    parser.add_argument("-s", "--submit", action = "store_true", help = "Default is False, use to start predicting.")

    args = parser.parse_args()

    print(f"{Color.blue}{' CONFIG '.center(int(get_terminal_length()), 'v')}{Color.reset}")
    print(f"{Color.purple}Precrawl: {f'{Color.green}True' if args.precrawl else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Crawl: {f'{Color.green}True' if args.crawl else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Preprocess: {f'{Color.green}True' if args.preprocess else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Training: {f'{Color.green}True' if args.train else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Predict: {f'{Color.green}True' if args.submit else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.blue}{'^'.center(int(get_terminal_length()), '^')}{Color.reset}")
    print("")

    # ^^ ======================================== PRECRAWL PROCESSING ======================================== ^^ #
    if args.precrawl:
        print(f"{Color.green}{Color.bold}{"    START PRECRAWLING    ".center(get_terminal_length(), "=")}{Color.reset}")

        df = pd.read_csv(config.TRAIN_SOURCE_PATH)
        df = precrawl_process(df, save_path=config.TRAIN_PRECRAWL_PATH)

        df = pd.read_csv(config.TEST_PUBLIC_SOURCE_PATH)
        df = precrawl_process(df, config.TEST_PUBLIC_PRECRAWL_PATH)

        df = pd.read_csv(config.TEST_PRIVATE_SOURCE_PATH)
        df = precrawl_process(df, config.TEST_PRIVATE_PRECRAWL_PATH)
        
        print(f"{Color.green}{Color.bold}{"    PRECRAWLING SUCCESS    ".center(get_terminal_length(), "=")}{Color.reset}")

    # ^^ ======================================== CRAWLING DATA ======================================== ^^ #
    if args.crawl:
        print(f"{Color.green}{Color.bold}{"    START CRAWLING    ".center(get_terminal_length(), "=")}{Color.reset}")

        df = pd.read_csv(config.TRAIN_PRECRAWL_PATH)
        crawl_data(df, save_path=config.TRAIN_POSTCRAWL_PATH, target="Label")

        df = pd.read_csv(config.TEST_PUBLIC_PRECRAWL_PATH)
        crawl_data(df, save_path=config.TEST_PUBLIC_POSTCRAWL_PATH)

        df = pd.read_csv(config.TEST_PRIVATE_PRECRAWL_PATH)
        crawl_data(df, save_path=config.TEST_PRIVATE_POSTCRAWL_PATH)

        print(f"{Color.green}{Color.bold}{"    CRAWLING SUCCESS    ".center(get_terminal_length(), "=")}{Color.reset}")

    # ^^ ======================================== PREPROCESSING DATA ======================================== ^^ #
    if args.preprocess:
        print(f"{Color.green}{Color.bold}{"    START PREPROCESSING    ".center(get_terminal_length(), "=")}{Color.reset}")

        df = pd.read_csv(config.TRAIN_POSTCRAWL_PATH)
        preprocess_data(df, save_path=config.TRAIN_PREPROCESS_PATH, target="Label")

        df = pd.read_csv(config.TEST_PUBLIC_POSTCRAWL_PATH)
        preprocess_data(df, save_path=config.TEST_PUBLIC_PREPROCESS_PATH)

        df = pd.read_csv(config.TEST_PRIVATE_POSTCRAWL_PATH)
        preprocess_data(df, save_path=config.TEST_PRIVATE_PREPROCESS_PATH)

        print(f"{Color.green}{Color.bold}{"    PREPROCESSING SUCCESS    ".center(get_terminal_length(), "=")}{Color.reset}")

    # ^^ ======================================== TRAINING MODEL ======================================== ^^ #
    if args.train:
        print(f"{Color.blue}{Color.bold}{'    START TRAINING    '.center(get_terminal_length(), '=')}{Color.reset}")
        train_model_specter_classifier(config.TRAIN_PREPROCESS_PATH, config.MODEL_PATH)
        print(f"{Color.green}{Color.bold}{'    TRAINING SUCCESS    '.center(get_terminal_length(), '=')}{Color.reset}")

    # ^^ ======================================== PREDICTION ======================================== ^^ #
    if args.submit:
        print(f"{Color.blue}{Color.bold}{'    START PREDICTING    '.center(get_terminal_length(), '=')}{Color.reset}")
        public_summission = predict(config.TEST_PUBLIC_PREPROCESS_PATH, config.SUBMISSION_PUBLIC_PATH)
        private_summission = predict(config.TEST_PRIVATE_PREPROCESS_PATH, config.SUBMISSION_PRIVATE_PATH)
        combine = pd.concat([public_summission, private_summission])
        combine.to_csv(config.COMBINE_SUBMISSION_PATH, index=False)
        print(f"{Color.green}{Color.bold}{'    PREDICTION SUCCESS    '.center(get_terminal_length(), '=')}{Color.reset}")
    

    winddown_msg = "    world.terminate(me)    ".center(get_terminal_length(), "=")
    print(f"{Color.pink}{Color.bold}{winddown_msg}{Color.reset}")