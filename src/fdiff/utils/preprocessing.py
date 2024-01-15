from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from tqdm import tqdm


def mimic_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in a MIMIC-III dataframe. Adapted from MIMIC-Extract https://shorturl.at/amtyQ.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to impute.

    Returns
    -------
    pd.DataFrame
        Imputed dataframe.
    """
    ID_COLS = ["subject_id", "hadm_id", "icustay_id"]

    idx = pd.IndexSlice
    df = df.copy()
    if len(df.columns.names) > 2:
        df.columns = df.columns.droplevel(("label", "LEVEL1", "LEVEL2"))

    # Only the mean features (avg measurement over 1 hour) shall be imputed.
    df_out = df.loc[:, idx[:, ["mean", "count"]]]  # type: ignore

    # Compute mean value over the whole stay for each patient.
    icustay_means = df_out.loc[:, idx[:, "mean"]].groupby(ID_COLS).mean()  # type: ignore

    # For each patient, propagate last observation forward if possible.
    # If not, fill with the mean value for that patient.
    # If no mean value is available, fill with 0.
    df_out.loc[:, idx[:, "mean"]] = (  # type: ignore
        df_out.loc[:, idx[:, "mean"]]  # type: ignore
        .groupby(ID_COLS)
        .ffill()
        .groupby(ID_COLS)
        .fillna(icustay_means)
        .fillna(0)
    )  # type: ignore

    # Create mask that highlights time steps at which no feature has been observed.
    df_out.loc[:, idx[:, "count"]] = (df.loc[:, idx[:, "count"]] > 0).astype(float)  # type: ignore
    df_out.rename(columns={"count": "mask"}, level="Aggregation Function", inplace=True)

    # When the feature is missing, compute the time since it was last measured.
    is_absent = 1 - df_out.loc[:, idx[:, "mask"]]  # type: ignore
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent == 0].ffill()
    time_since_measured.rename(
        columns={"mask": "time_since_measured"},
        level="Aggregation Function",
        inplace=True,
    )

    # Add the time since the last measurement to the dataframe.
    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, "time_since_measured"]] = df_out.loc[  # type: ignore
        :, idx[:, "time_since_measured"]  # type: ignore
    ].fillna(100)

    # Return a dataframe with sorted columns.
    df_out.sort_index(axis=1, inplace=True)
    return df_out


def mimic_to_3D_tensor(df: pd.DataFrame) -> np.ndarray:
    idx = pd.IndexSlice
    return np.dstack(
        [
            df.loc[idx[:, :, :, i], :].values
            for i in sorted(set(df.index.get_level_values("hours_in")))
        ]
    )


def mimic_preprocess(data_dir: Path, random_seed: int, train_frac: float = 0.8) -> None:
    """Preprocess the MIMIC-III dataset from the raw h5 file in data_dir.
    Saves the preprocessed data as .pt files in the same directory.

    Args:
        data_dir (Path): Path in which the raw "all_hourly_data.h5" file is stored.
    """
    dataset_path = data_dir / "all_hourly_data.h5"
    GAP_TIME = 6  # In hours
    WINDOW_SIZE = 24  # In hours

    # Load the static and dynamic dataframes
    statics = pd.read_hdf(dataset_path, "patients")
    df = pd.read_hdf(dataset_path, "vitals_labs")

    # Extract the target
    Ys = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME][
        ["mort_hosp", "mort_icu", "los_icu"]
    ]
    Ys["los_3"] = Ys["los_icu"] > 3
    Ys["los_7"] = Ys["los_icu"] > 7
    Ys.drop(columns=["los_icu"], inplace=True)
    Ys.astype(float)

    # Extract the rows of the dynamic dataframe that have a corresponding target
    # and that correspond to the first 24 hours of stay.
    lvl2 = df[
        (
            df.index.get_level_values("icustay_id").isin(
                set(Ys.index.get_level_values("icustay_id"))
            )
        )
        & (df.index.get_level_values("hours_in") < WINDOW_SIZE)
    ]

    # Extract the identifiers of all patients in the datasets.
    lvl2_subj_idx, Ys_subj_idx = [
        df.index.get_level_values("subject_id") for df in (lvl2, Ys)
    ]
    lvl2_subjects = set(lvl2_subj_idx)
    assert lvl2_subjects == set(Ys_subj_idx), "Subject ID pools differ!"

    # Split the dataset into train and test patients
    assert 0 < train_frac < 1, f"train_frac must be in (0, 1), but got {train_frac=}."
    np.random.seed(random_seed)
    subjects, N = np.random.permutation(list(lvl2_subjects)), len(lvl2_subjects)
    N_train = int(train_frac * N)
    train_subj = subjects[:N_train]
    test_subj = subjects[N_train:]
    (lvl2_train, lvl2_test) = [
        lvl2[lvl2.index.get_level_values("subject_id").isin(s)]
        for s in (train_subj, test_subj)
    ]

    # Standardize each feature over all times and patients.
    idx = pd.IndexSlice
    lvl2_means, lvl2_stds = lvl2_train.loc[:, idx[:, "mean"]].mean(  # type: ignore
        axis=0
    ), lvl2_train.loc[
        :, idx[:, "mean"]  # type: ignore
    ].std(  # type: ignore
        axis=0
    )  # type: ignore
    lvl2_train.loc[:, idx[:, "mean"]] = (  # type: ignore
        lvl2_train.loc[:, idx[:, "mean"]] - lvl2_means  # type: ignore
    ) / lvl2_stds
    lvl2_test.loc[:, idx[:, "mean"]] = (  # type: ignore
        lvl2_test.loc[:, idx[:, "mean"]] - lvl2_means  # type: ignore
    ) / lvl2_stds
    assert isinstance(lvl2_train, pd.DataFrame) and isinstance(lvl2_test, pd.DataFrame)

    # Impute the missing values.
    lvl2_train, lvl2_test = [mimic_imputer(df) for df in (lvl2_train, lvl2_test)]

    # Check that there is no missing value after imputation.
    for df in lvl2_train, lvl2_test:
        assert not df.isnull().any().any()

    # Convert the train and test dataframes to 3D tensors.
    X_train, X_test = [
        rearrange(
            torch.from_numpy(
                mimic_to_3D_tensor(df.loc[:, pd.IndexSlice[:, "mean"]]).astype(  # type: ignore
                    np.float32
                )
            ),
            "example_idx channel time -> example_idx time channel",
        )
        for df in (lvl2_train, lvl2_test)
    ]

    # Check that each time series has 24 time steps and 104 features
    for X in X_train, X_test:
        assert X.size() == (len(X), 24, 104)

    # Save the preprocessed tensors.
    for X, name in zip([X_train, X_test], ["train", "test"]):
        torch.save(X, data_dir / f"X_{name}.pt")


def nasdaq_preprocess(
    data_dir: Path,
    random_seed: int,
    train_frac: float = 0.9,
    start_date: str = "2019-01-01",
    end_date: str = "2020-01-01",
) -> None:
    """Preprocess the NASDAQ dataset from the raw .csv file in data_dir.
    Saves the preprocessed data as .pt files in the same directory.

    Args:
        data_dir (Path): Path in which the raw financial data is stored.
    """

    # Collate all stock data into a single dataframe
    df_list = []
    stock_paths = list((data_dir / "stocks").glob("*.csv"))
    for path in tqdm(
        stock_paths, unit="stock", leave=False, desc="Loading stock data", colour="blue"
    ):
        df_stock = pd.read_csv(path)
        df_stock["Name"] = path.stem.strip(".csv")
        df_list.append(df_stock)
    df = pd.concat(df_list, axis=0, ignore_index=True)

    # Check that all the stocks have been recorded
    assert len(df["Name"].unique()) == len(stock_paths)

    # Convert date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    start_time = pd.to_datetime(start_date)
    end_time = pd.to_datetime(end_date)

    # Remove stocks that are not active in the whole time interval
    stocks_older_than_start = set()
    stocks_active_in_end = set()
    for stock, valid in (df.groupby("Name")["Date"].min() <= start_time).items():
        if valid:
            stocks_older_than_start.add(stock)
    for stock, valid in (df.groupby("Name")["Date"].max() >= end_time).items():
        if valid:
            stocks_active_in_end.add(stock)
    valid_stocks = stocks_older_than_start.intersection(stocks_active_in_end)
    df = df[
        df["Name"].isin(valid_stocks)
        & (df["Date"] >= start_time)
        & (df["Date"] < end_time)
    ]

    # Remove stocks with missing days
    stocks_no_missing_day = set()
    for stock, valid in (df.groupby("Name")["Date"].nunique() == 252).items():
        if valid:
            stocks_no_missing_day.add(stock)
    df = df[df["Name"].isin(stocks_no_missing_day)]

    # Create a tensor of shape (num_stocks, num_days, num_features) from df
    df_pivot = df.pivot_table(
        index="Name",
        columns="Date",
        values=["Open", "High", "Low", "Close", "Adj Close", "Volume"],  # type: ignore
    )
    X = torch.tensor(df_pivot.values, dtype=torch.float32)
    X = rearrange(X, "stock (feature day) -> stock day feature", day=252)

    # Train-test split
    torch.manual_seed(random_seed)
    num_train = int(train_frac * len(X))
    perm_idx = torch.randperm(len(X))
    train_stocks, test_stocks = perm_idx[:num_train], perm_idx[num_train:]
    X_train, X_test = X[train_stocks], X[test_stocks]

    # Save the preprocessed tensors.
    for X, name in zip([X_train, X_test], ["train", "test"]):
        torch.save(X, data_dir / f"X_{name}.pt")


def nasa_preprocess(
    data_dir: Path,
    subdataset: str = "charge",
    train_frac: float = 0.9,
    random_seed: int = 42,
) -> None:
    if subdataset == "charge":
        features = [
            "Voltage_measured",
            "Current_measured",
            "Temperature_measured",
            "Current_charge",
            "Voltage_charge",
        ]
        sub_dataset = "charge"
        interval_bin = 10
        cutoff_time = 5000 - 5000 % interval_bin
    elif subdataset == "discharge":
        features = [
            "Voltage_measured",
            "Current_measured",
            "Temperature_measured",
            "Current_load",
            "Voltage_load",
        ]
        sub_dataset = "discharge"
        interval_bin = 15
        cutoff_time = 2000 - 2000 % interval_bin

    else:
        raise ValueError(f"Unknown subdataset {subdataset}")

    # Read the metadata
    metadata = pd.read_csv(data_dir / "cleaned_dataset" / "metadata.csv")
    files = metadata[metadata["type"] == f"{sub_dataset}"]["filename"].values

    full_df = pd.DataFrame()

    for filename in tqdm(files):
        data = pd.read_csv(data_dir / "cleaned_dataset" / "data" / filename)

        # check if the maximum time is greater than the cutoff time
        if data["Time"].max() > cutoff_time:
            # Check that the maximum interval is less than the bin size
            interval = data["Time"].diff().max()
            if interval > interval_bin:
                continue

            # Remove the rows such that the time is greater than the cutoff time
            data = data[data["Time"] < cutoff_time]

            # bin the data

            data["Time_Bin"] = pd.cut(
                data["Time"],
                bins=range(
                    -interval_bin, int(cutoff_time + interval_bin), interval_bin
                ),
            )

            # Group by custom bins and calculate the mean for each group
            result_df = data.groupby("Time_Bin", observed=False).mean().reset_index()
            result_df["Time_Bin"] = result_df.index

            result_df["filename"] = filename
            full_df = pd.concat([full_df, result_df])

    df_pivot = full_df.pivot(index="filename", columns="Time_Bin", values=features)

    num_timesteps = cutoff_time // interval_bin + 1
    X_full = torch.tensor(df_pivot.values, dtype=torch.float32)
    # Rearange to get a 3D tensor of shape (num_samples, num_timesteps, num_features)
    X_reshaped = X_full.reshape(X_full.shape[0], -1, num_timesteps)
    # Permute the last two dimensions
    X = X_reshaped.permute(0, 2, 1)

    # Train-test split
    torch.manual_seed(random_seed)
    num_train = int(train_frac * len(X))
    perm_idx = torch.randperm(len(X))
    train_stocks, test_stocks = perm_idx[:num_train], perm_idx[num_train:]
    X_train, X_test = X[train_stocks], X[test_stocks]

    # create the directory if it does not exist
    folder = data_dir / subdataset
    folder.mkdir(parents=True, exist_ok=True)

    # Save the preprocessed tensors.
    for X, name in zip([X_train, X_test], ["train", "test"]):
        torch.save(X, data_dir / subdataset / f"X_{name}.pt")


def droughts_preprocess(
    data_dir: Path,
    random_seed: int,
    train_frac: float = 0.9,
    start_date: str = "2011-01-01",
    end_date: str = "2012-01-01",
) -> None:
    """Preprocess the US-Droughts dataset from the raw .csv file in data_dir.
    Saves the preprocessed data as .pt files in the same directory.

    Args:
        data_dir (Path): Path in which the raw financial data is stored.
    """

    df = pd.read_csv(data_dir / "train_timeseries/train_timeseries.csv")

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Remove entries that are not in the inderval
    start_time = pd.to_datetime(start_date)
    end_time = pd.to_datetime(end_date)
    df = df[(df["date"] >= start_time) & (df["date"] < end_time)]

    # Drop columns with missing values
    df = df.dropna(axis=1)

    # Create a tensor of shape (num_fips, num_days, num_features) from df
    df_pivot = df.pivot_table(index="fips", columns="date")
    num_days = (end_time - start_time).days
    X = torch.tensor(df_pivot.values, dtype=torch.float32)
    X = rearrange(X, "fips (feature day) -> fips day feature", day=num_days)

    # Train-test split
    torch.manual_seed(random_seed)
    num_train = int(train_frac * len(X))
    perm_idx = torch.randperm(len(X))
    train_fips, test_fips = perm_idx[:num_train], perm_idx[num_train:]
    X_train, X_test = X[train_fips], X[test_fips]

    # Save the preprocessed tensors.
    for X, name in zip([X_train, X_test], ["train", "test"]):
        torch.save(X, data_dir / f"X_{name}.pt")
