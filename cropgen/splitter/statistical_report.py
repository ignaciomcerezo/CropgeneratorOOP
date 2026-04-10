import re

import pandas as pd

from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface

_columns_to_stratify_with = [
    "paragraph",
    "order",
    "is_letter",
]

# columnas sobre las que calculamos las estadísticas
_columns_to_use_categorical = [
    "paragraph",
    "order",
    "is_letter",
]

_columns_to_use_numerical = [
    "sindex",
    "average_rotation",
    "text_length",
    "math_percentage",
]

_columns_to_use = _columns_to_use_categorical + _columns_to_use_numerical

math_pattern = re.compile(r"(?<!\\)(\$\$?)(.*?)(?<!\\)\1", re.DOTALL)


class PairsStatisticalData:
    def __init__(self, pdi: PairsDataInterface):
        df = pdi.df.copy()
        df["text_length"] = df.text.apply(len)
        df["math_percentage"] = df.text.apply(math_percentage)

        self.df = df

        self.paragraph = self._describe_categorical(df.paragraph)
        self.is_letter = self._describe_categorical(df.is_letter)
        self.order = self._describe_categorical(df.order)
        self.is_letter = self._describe_categorical(df.is_letter)

        self.sindex = self._describe_numerical(df.sindex)
        self.text_length = self._describe_numerical(df.text_length)
        self.average_rotation = self._describe_numerical(df.average_rotation)
        self.math_percentage = self._describe_numerical(df.math_percentage)

        self.stratified: dict[str, dict[str, pd.DataFrame]] = {}
        for col_one in _columns_to_use:
            self.stratified[col_one] = {}
            for col_other in _columns_to_stratify_with:
                self.stratified[col_one][col_other] = self.stratify_one_by_other(
                    df, col_one, col_other
                )

    @staticmethod
    def _describe_numerical(column: pd.DataFrame):
        return pd.DataFrame.from_dict(
            {
                "mean": [column.mean()],
                "min": [column.min()],
                "max": [column.max()],
                "std": [column.std()],
            }
        )

    @staticmethod
    def _describe_categorical(column: pd.Series):
        return pd.DataFrame(column.describe())

    @staticmethod
    def stratify_one_by_other(
        df: pd.DataFrame, col_one: str, col_other: str
    ) -> pd.DataFrame:
        # todo: WIP
        assert col_other in _columns_to_stratify_with
        assert col_one in _columns_to_use

        des_func = (
            PairsStatisticalData._describe_categorical
            if col_one in _columns_to_use_categorical
            else PairsStatisticalData._describe_numerical
        )

        values_strata = sorted([str(x) for x in pd.unique(df[col_other])])

        dfs = []
        for value in values_strata:
            desc = des_func(df[df[col_other] == value][col_one])
            desc["values_strata"] = value
            dfs.append(desc)
        df_stratified_desc = pd.concat(dfs)
        df_stratified_desc = df_stratified_desc.set_index("values_strata")

        return df_stratified_desc


def count_tokens(text: str) -> int:
    r"""
    Estimates the nubmer of tokens in a string, matching latex commands, standards words and individual
    punctuation and symbols.
    """
    tokens = re.findall(r"\\[a-zA-Z]+|\w+|[^\w\s]", text)
    return len(tokens)


def math_percentage(text: str) -> float:

    clean_text = text.replace(r"\$", "")

    if clean_text.count("$") % 2 != 0:
        return -1.0
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", text, flags=re.DOTALL)

    math_tokens = 0
    text_tokens = 0

    for i, part in enumerate(parts):
        if not part.strip():
            continue

        if i % 2 == 1:
            math_content = part.strip("$")
            math_tokens += count_tokens(math_content)
        else:
            text_tokens += count_tokens(part)

    total_tokens = math_tokens + text_tokens

    if total_tokens == 0:
        return 0.0

    return math_tokens / total_tokens
