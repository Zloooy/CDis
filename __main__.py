from polars import DataFrame, Series, read_csv, col, struct, lit, max as plmax, when
from itertools import product, starmap, pairwise
from functools import reduce
from operator import itemgetter


def prepare_table(df: DataFrame, input_var_cols: list[str], output_var_cols: list[str]):
    output_var_mappings = []
    for col_name in input_var_cols:
        output_var_mappings.append(
            tuple(sorted(df.select(col(col_name).unique()).to_dict()[col_name]))
        )
    map_dict = dict(
        starmap(lambda i, prod: (prod, i), enumerate(product(*output_var_mappings)))
    )

    def series_to_int(s: Series):
        return map_dict[tuple(s.values())]

    return df.with_columns(
        struct(output_var_cols).apply(series_to_int).alias("output"),
        lit(False).alias("filtered"),
        lit(0).alias("order"),
    )


def indiscernibility_partition(df, input_var_cols):
    order = list(range(df.height))
    for col_name in input_var_cols:
        temp_order = order.copy()
        value_counts = list(
            map(
                itemgetter(1),
                sorted(
                    map(
                        lambda v: (v[col_name], v["counts"]),
                        df.select(col(col_name).value_counts())
                        .to_dict()[col_name]
                        .to_list(),
                    ),
                    key=itemgetter(0),
                ),
            )
        )
        for k in range(1, len(value_counts)):
            value_counts[k] += value_counts[k - 1]
        for j, c in zip(
            reversed(temp_order),
            df.select(col(col_name).where(col("filtered") == False))
            .reverse()
            .to_dict(as_series=False)[col_name],
        ):
            order[value_counts[c] - 1] = j
            value_counts[c] -= 1

    def reducer(acc, x):
        if len(acc) == 0 or any(
            map(
                lambda col_name: x[1][col_name] != acc[-1][-1][1][col_name],
                input_var_cols,
            )
        ):
            return (*acc, [x])
        else:
            return (*acc[:-1], [*acc[-1], x])

    return list(
        map(
            lambda group: tuple(map(itemgetter(0), group)),
            reduce(reducer, map(lambda i: (i, df.row(i, named=True)), order), []),
        )
    )


def simplified_decision_table(df, indiscernibility_parts):
    res_group = []
    max_output = df.select(plmax("output"))
    for group in indiscernibility_parts:
        rows = list(map(lambda i: df.row(i, named=True), group))
        if len(set(map(itemgetter("output"), rows))) > 1:
            df.with_row_count("row_nr").select(
                when(col("row_nr") == group[0]).then(max_output + 1).alias("output")
            )
            for i in rows[1:]:
                df.with_row_count("row_nr").select(
                    when(col("row_nr") == group[i]).then(True).alias("blocked")
                )
            df.with_row_count("row_nr").select(
                when(col("row_nr") == group[0]).then(max_output + 1)
            )
        res_group.append(group[0])
    return res_group


def positive_region(df, decision_table):
    # TODO implement
    pass


def main():
    table_path = "./solution_table.csv"
    input_var_cols = list(map(lambda i: f"i{i}", range(10)))
    output_var_cols = list(map(lambda i: f"o{i}", range(10)))
    df: DataFrame = read_csv(table_path)
    df = prepare_table(df, input_var_cols, output_var_cols)
    indiscernibility_parts = indiscernibility_partition(df, input_var_cols)
    simple_table = simplified_decision_table(df, indiscernibility_parts)
    print(simple_table)


if __name__ == "__main__":
    main()
