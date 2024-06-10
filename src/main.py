import data_processing.process_and_merge_csv as process_and_merge_csv
import data_processing.data_science_fundamentals as data_science_fundamentals

def main():
    process_and_merge_csv.process_files()
    data_science_fundamentals.apply_measures()

if __name__ == "__main__":
    main()
