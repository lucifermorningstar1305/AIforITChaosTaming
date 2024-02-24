import numpy as np
import polars as pol
import argparse
import time
import os
import logging

from dotenv import load_dotenv
from langchain_community.chat_models import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

load_dotenv()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", type=str, required=True, help="the path of the few shot dataset."
    )

    parser.add_argument(
        "--n_examples",
        "-n",
        type=int,
        required=False,
        default=5,
        help="the number of examples of each class to sample for few-shot prompting.",
    )

    args = parser.parse_args()

    data_path = args.data_path
    n_examples = args.n_examples

    if not os.path.exists("../logs"):
        os.mkdir("../logs")

    logging.basicConfig(
        filename="../logs/genai.log",
        filemode="w+",
        format="%(asctime)s-%(levelname)s-%(message)s",
        level=logging.DEBUG,
    )

    df = pol.read_csv(data_path)

    unique_labels = df["Assigned_Group_fixed"].unique().to_list()

    clean_texts = list()
    labels = list()

    for label in unique_labels:
        _sampled_data = df.filter(pol.col("Assigned_Group_fixed") == label).sample(
            n=n_examples, shuffle=True, with_replacement=True, seed=32
        )
        clean_text = _sampled_data["clean_text"].to_list()
        clean_texts.append(clean_text)

        grps = _sampled_data["Assigned_Group_fixed"].to_list()
        labels.extend(grps)

    examples = list()

    for email, label in zip(clean_text, labels):
        examples.append({"email": email, "label": label})

    df_new = df.filter(~pol.col("clean_text").is_in(clean_text))

    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{email}"), ("ai", "{label}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt, examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            few_shot_prompt,
            ("human", "{email}"),
        ]
    )

    chain = final_prompt | ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GEMINI_KEY"),
        temperature=0.0,
        convert_system_message_to_human=True,
    )

    # chain = final_prompt | ChatOpenAI(
    #     temperature=0, openai_api_key=os.getenv("OPENAI_KEY")
    # )

    # ans = chain.invoke({"email": df_new["clean_text"].to_list()[0]})
    # print(type(ans), ans.content, df_new["Assigned_Group_fixed"].to_list()[0])

    n_correct = 0
    n_records = 0

    if os.path.exists('./status.txt'): 
        with open("./status.txt", "r") as fp:
            status = fp.readline()

        status = list(map(lambda x: int(x), status.split()))
        df_new = df_new.slice(status[0])
        n_correct = status[1]
        n_records = status[2]

    for idx, row in enumerate(df_new.rows(named=True)):

        try:
            email = row["clean_text"]
            label = row["Assigned_Group_fixed"]

            llm_ans = chain.invoke({"email": email})
            if llm_ans.content == label:
                n_correct += 1

            if idx % 60 == 0 and idx != 0:
                with open("./status.txt", "w") as fp:
                    fp.write(f"{idx} {n_correct} {n_records}")
                
                time.sleep(60)

            n_records += 1
            print(f"Number of records parsed: {n_records}, Number of correct: {n_correct}")
        except Exception as e:
            logging.error(e)

    print(f"Number of correct : {n_correct} / {n_records}")

    if os.path.exists("./status.txt"):
        os.remove("./status.txt")
