import re
from cropgen.splitter.crops_interface.PairsDataInterface import PairsDataInterface
import pandas as pd

_columns_to_stratify_with = [
    "paragraph",
    "order",
    "is_letter",
]

# columnas sobre las que calculamos las estadísticas
_columns_to_use_categorical = [
    "paragraph",
    "order",
    "has_enough_context",
    "is_letter",
]

_columns_to_use_numerical = [
    "sindex",
    "text",
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

        self.paragraph = self._describe_categorical(df.paragraph)
        self.is_letter = self._describe_categorical(df.is_letter)
        self.order = self._describe_categorical(df.order)
        self.has_enough_context = self._describe_categorical(df.has_enough_context)
        self.is_letter = self._describe_categorical(df.is_letter)

        self.sindex = self._describe_numerical(df.sindex)
        self.text_length = self._describe_numerical(df.text_length)
        self.average_rotation = self._describe_numerical(df.average_rotation)
        self.math_percentage = self._describe_numerical(df.math_percentage)

        self.stratified = {}
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
        return column.describe()

    @staticmethod
    def stratify_one_by_other(
        df: pd.DataFrame, col_one: str, col_other: str
    ) -> pd.DataFrame:
        assert col_other in _columns_to_stratify_with
        assert col_one in _columns_to_use

        des_func = (
            PairsStatisticalData._describe_categorical
            if col_one in _columns_to_use_categorical
            else PairsStatisticalData._describe_numerical
        )

        values_strata = sorted(pd.unique(df[col_other]))

        df_stratified_desc = des_func(df[df[col_other] == values_strata[0]][col_one])

        for value in values_strata[1:]:
            df_stratified_desc = pd.concat(
                [df_stratified_desc, des_func(df[df[col_other] == value][col_one])]
            )
        return df_stratified_desc


def text_inside_math(text: str) -> str:
    return extract_math_from_dollars(text)


def text_outside_math(text: str) -> str:
    return extract_math_from_dollars("$" + text + "$")


def math_percentage(text: str) -> float:
    if text.replace(r"\$", "").count("$") % 2:  # hay "$" desparejados
        return -1

    in_math_length = len(text_inside_math(text))
    out_math_length = len(text_inside_math(text))

    if not in_math_length and not out_math_length:
        return 0
    return in_math_length / (out_math_length + in_math_length)


def extract_math_from_dollars(text, separator=" "):

    matches = math_pattern.findall(text)
    extracted_text = [match[1].strip() for match in matches]

    return separator.join(extracted_text)
