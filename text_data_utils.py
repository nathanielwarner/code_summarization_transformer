import re
import csv
import json
from typing import Tuple, List


def rephrase_question(x: str) -> str:
    if len(x) < 2:
        return x
    if x[-1] == '?':
        x = x[:-1]
    for opener in ['how do i ', 'how do you ', 'how can i ', 'how to ', 'best way to ', 'can i ',
                   'is there a way to ', 'easiest way to ', 'best implementation for ',
                   'best implementation of ', 'what is the best way to ', 'what is the proper way to ',
                   'is it possible to ', 'would it be possible to '
                   'how ', 'c# how to ', 'c# how ', 'c# - ', 'c# ']:
        if x.lower().startswith(opener):
            x = x[len(opener):]
    for closer in [' in c#', ' with c#', ' using c#', ' c#']:
        if x.lower().endswith(closer):
            x = x[:-len(closer)]
    return x


def preprocess(x: str, remove_stars=False, remove_java_doc_vars=False, remove_html_tags=False, remove_comments=False,
               remove_start_and_end_quotes=False, rephrase=False, lower=False, to_edinburgh_format=False) -> str:
    if to_edinburgh_format:
        if x.endswith('\n'):
            x = x[:-len('\n')]
        x = x.replace('\n', ' DCNL ')
        x = x.replace('    ', ' DCSP ')
        x = x.replace('\t', ' DCSP ')
    if remove_java_doc_vars:
        x = re.sub(r'(?<![{])(@[\s\S]*)', ' ', x)
    if remove_comments:
        x = re.sub(r'(?<![:\"])(//.*?(?:\n|\\n))', ' ', x)
    if remove_html_tags:
        x = re.sub(r'<.*?>', ' ', x)
    x = x.replace('\\n', ' ').replace('\n', ' ')
    x = x.replace('\\t', ' ').replace('\t', ' ')
    if remove_stars:
        x = x.replace('/*', ' ').replace('*/', ' ').replace('*', ' ')
    if remove_start_and_end_quotes:
        x = x.strip()
        if x.startswith("'"):
            x = x[len("'"):]
        if x.endswith("'"):
            x = x[:-len("'")]
        if x.startswith('"'):
            x = x[len('"'):]
        if x.endswith('"'):
            x = x[:-len('"')]
    x = x.strip()
    x = re.sub(r'(\s\s+)', ' ', x)
    if rephrase:
        x = rephrase_question(x)
    if lower:
        x = x.lower()
    return x


def preprocess_java(x: str) -> str:
    return preprocess(x, remove_comments=True, remove_start_and_end_quotes=True)


def preprocess_javadoc(x: str) -> str:
    return preprocess(x, remove_stars=True, remove_java_doc_vars=True, remove_html_tags=True)


def json_java_dataset_as_generator(filename):
    file = open(filename, mode='r', encoding='utf-8')

    def generator():
        while True:
            row = file.readline()
            if len(row) == 0:
                file.seek(0)
                break
            json_row = json.loads(row)
            summary = preprocess_javadoc(json_row["nl"])
            code = preprocess_java(json_row["code"])
            yield summary, code

    return generator


def load_json_dataset(filename):
    generator = json_java_dataset_as_generator(filename)
    return list(generator())


def generator_from_list(rows):
    def generator():
        for row in rows:
            yield row
    return generator


def eof_text(text: str) -> str:
    text = "<s>" + text + "</s>"
    return text


def de_eof_text(text: str) -> str:
    if text.startswith("<s>"):
        text = text[len("<s>"):]
    if "</s>" in text:
        text = text[:text.index("</s>")]
    return text
