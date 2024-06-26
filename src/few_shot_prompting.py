import numpy as np
import polars as pol
import argparse
import time
import os
import logging
import json

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

    print(unique_labels)

    clean_texts = list()
    labels = list()

    for label in unique_labels:
        _sampled_data = df.filter(pol.col("Assigned_Group_fixed") == label).sample(
            n=n_examples, shuffle=True, with_replacement=True, seed=32
        )
        clean_text = _sampled_data["clean_text"].to_list()
        clean_texts.extend(clean_text)

        grps = _sampled_data["Assigned_Group_fixed"].to_list()
        labels.extend(grps)

    examples = list()

    for email, label in zip(clean_texts, labels):
        examples.append({"input": email, "output": label})

    df_new = df.filter(~pol.col("clean_text").is_in(clean_text))

    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt, examples=examples
    )

    print(few_shot_prompt.format())
    exit(0)

    print("\n" * 2)
    print("-" * 20)
    print("\n")

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"Predict the labels for the corresponding emails. The output should be either one of the following labels {unique_labels}",
            ),
            few_shot_prompt,
            ("human", "{input}"),
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

    response = list()

    if not os.path.exists("../responses"):
        os.mkdir("../responses")

    for idx, row in enumerate(df_new.rows(named=True)):

        _temp_res = dict()

        try:
            _id = row["Incident_No"]
            email = row["clean_text"]
            label = row["Assigned_Group_fixed"]
            _temp_res["id"] = _id
            _temp_res["email"] = email
            _temp_res["label"] = label
            llm_ans = chain.invoke({"input": email})
            _temp_res["llm_response"] = llm_ans.content

            print(f"{idx} | Incident: {_id} Label: {label} Pred: {llm_ans.content}")

            response.append(_temp_res)

            if idx % 60 == 0 and idx != 0:
                time.sleep(60)

        except Exception as e:
            logging.error(e)

    if os.path.exists("./status.txt"):
        os.remove("./status.txt")

    with open(
        f"../responses/response_{n_examples * len(unique_labels)}_shot.json", "w"
    ) as fp:
        json.dump(response, fp)
