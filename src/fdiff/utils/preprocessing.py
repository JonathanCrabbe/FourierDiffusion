from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange


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
        data_dir (Path): Path in which the raw "all_hourly_data.h5 file is stored.
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
