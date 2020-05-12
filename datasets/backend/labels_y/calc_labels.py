# General:
import os
import pandas as pd

# Path variable:
datasets_dr = '../'


def get_labels():
    """
    1. Retrieves experimental and calculated hydration free energies with associated uncertainties
    from the FreeSolve database.
    2. Offsets between experimental and calculated are calculated.
    3. Offsets DataFrame and experimental hydration free energies (control) are written to CSV.
    """

    # List of columns names to be read from FreeSolv
    cols = ['compound id (and file prefix)',                        # index
            'experimental value (kcal/mol)',                        # 0
            'experimental uncertainty (kcal/mol)',                  # 1
            'Mobley group calculated value (GAFF) (kcal/mol)',      # 2
            'calculated uncertainty (kcal/mol)']                    # 3

    # Load experimental and calculated dGhydr from FreeSolve database, sorted according to Mobley IDs.
    fs_df = pd.read_csv(filepath_or_buffer=datasets_dr + 'freesolv_database.txt', sep='; ', engine='python',
                        usecols=cols, index_col='compound id (and file prefix)').sort_index()

    # Offset between experimental and calculated hydration free energies.
    offset = pd.DataFrame.from_dict(
        data={name: [row[0] - row[2], sum_error(row[1], row[3])] for name, row in fs_df.iterrows()},
        orient='index',
        columns=['dGoffset (kcal/mol)', 'uncertainty (kcal/mol)']
    )

    # Experimental hydration free energies only as control data set.
    dGhydr = pd.DataFrame.from_dict(
        data={name: [row[0], row[1]] for name, row in fs_df.iterrows()},
        orient='index',
        columns=['Experimental dGhydr (kcal/mol)', 'uncertainty (kcal/mol)']
    )
    # need to drop some mols to exclude:
    excluded = ["mobley_6309289", "mobley_3395921", "mobley_6739648", "mobley_2607611", "mobley_637522", "mobley_172879"]
    for mol in excluded:
        offset.drop(mol, inplace=True)
        dGhydr.drop(mol, inplace=True)

    # Save to CSV.
    print(offset)
    save_csv(dataframe=offset, pathname='experimental_labels.csv')
    save_csv(dataframe=dGhydr, pathname='null_experimental_labels.csv')

    return offset, dGhydr


def sum_error(error1, error2):
    """Returns sum propagated error between two values."""
    return (error1 ** 2 + error2 ** 2) ** 0.5


def save_csv(dataframe, pathname):
    """Saves or overwrites DataFrame to CSV at given pathname,"""

    if os.path.exists(pathname):
        os.remove(pathname)
        dataframe.to_csv(path_or_buf=pathname, index=True, index_label='ID')
        print('Existing {} overwritten.'.format(pathname))
    else:
        dataframe.to_csv(path_or_buf=pathname, index=True)
        print('{} written.'.format(pathname))


if __name__ == '__main__':
    get_labels()
