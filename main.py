import pandas as pd
import time
import logging
import sys
from typing import List, Dict, Any
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from dataclasses import dataclass
from pathlib import Path

# Configuration class
@dataclass
class Config:
    csv_file: str = 'nyc_taxi_trip_duration.csv'
    data_splits: Dict[str, float] = None
    log_level: str = 'INFO'

    def __post_init__(self):
        if self.data_splits is None:
            self.data_splits = {
                '25%': 0.25,
                '50%': 0.50,
                '75%': 0.75,
                '100%': 1.0
            }

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('processing.log')
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to handle data processing operations with parallel processing capabilities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.num_workers = cpu_count()
        logger.info(f"Initialized DataProcessor with {self.num_workers} CPU cores")

    def _validate_data(self, data: List[float]) -> None:
        """Validate input data."""
        if not data:
            raise ValueError("Input data cannot be empty")
        if not all(isinstance(x, (int, float)) for x in data):
            raise ValueError("All data elements must be numbers")

    @staticmethod
    def sort_chunk(chunk: List[float]) -> List[float]:
        """Sort a single chunk of data."""
        return sorted(chunk)

    @staticmethod
    def filter_chunk(chunk: List[float]) -> List[float]:
        """Filter a single chunk of data."""
        return [x for x in chunk if x > 1000]

    def sort_sequential(self, data: List[float]) -> List[float]:
        """Sort data using a single process."""
        return sorted(data)

    def sort_threaded(self, data: List[float], num_threads: int) -> List[float]:
        """Sort data using multiple threads."""
        if num_threads <= 0:
            return self.sort_sequential(data)

        chunk_size = max(1, len(data) // num_threads)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        with ThreadPool(num_threads) as pool:
            sorted_chunks = pool.map(self.sort_chunk, chunks)

        return sorted([item for sublist in sorted_chunks for item in sublist])

    def sort_multiprocessing(self, data: List[float], num_processes: int) -> List[float]:
        """Sort data using multiple processes."""
        if num_processes <= 0:
            return self.sort_sequential(data)

        chunk_size = max(1, len(data) // num_processes)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        with Pool(num_processes) as pool:
            sorted_chunks = pool.map(self.sort_chunk, chunks)

        return sorted([item for sublist in sorted_chunks for item in sublist])

    def filter_sequential(self, data: List[float]) -> List[float]:
        """Filter data using a single process."""
        return self.filter_chunk(data)

    def filter_threaded(self, data: List[float], num_threads: int) -> List[float]:
        """Filter data using multiple threads."""
        if num_threads <= 0:
            return self.filter_sequential(data)

        chunk_size = max(1, len(data) // num_threads)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        with ThreadPool(num_threads) as pool:
            return [item for sublist in pool.map(self.filter_chunk, chunks) for item in sublist]

    def filter_multiprocessing(self, data: List[float], num_processes: int) -> List[float]:
        """Filter data using multiple processes."""
        if num_processes <= 0:
            return self.filter_sequential(data)

        chunk_size = max(1, len(data) // num_processes)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        with Pool(num_processes) as pool:
            return [item for sublist in pool.map(self.filter_chunk, chunks) for item in sublist]

    def load_data(self) -> List[float]:
        """Load and validate data from CSV file."""
        try:
            file_path = Path(self.config.csv_file)
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            data = df['trip_duration'].dropna().tolist()
            self._validate_data(data)
            logger.info(f"Successfully loaded {len(data):,} rows from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def run_analysis(self) -> pd.DataFrame:
        """Run performance analysis on different data sizes."""
        try:
            data = self.load_data()
            results = []

            for name, fraction in self.config.data_splits.items():
                logger.info(f"Analyzing data size: {name} ({int(len(data) * fraction):,} records)")
                data_subset = data[:int(len(data) * fraction)]

                # Measure sorting performance
                timings = {}
                for method, func in [
                    ('Sort Sequential', self.sort_sequential),
                    ('Sort Threading', lambda x: self.sort_threaded(x, self.num_workers)),
                    ('Sort Multiprocessing', lambda x: self.sort_multiprocessing(x, self.num_workers)),
                    ('Filter Sequential', self.filter_sequential),
                    ('Filter Threading', lambda x: self.filter_threaded(x, self.num_workers)),
                    ('Filter Multiprocessing', lambda x: self.filter_multiprocessing(x, self.num_workers))
                ]:
                    start_time = time.time()
                    func(data_subset.copy())
                    timings[f"{method} (s)"] = time.time() - start_time

                results.append({"Data Size": name, **timings})

            df_results = pd.DataFrame(results)
            logger.info("\n--- FINAL PERFORMANCE ANALYSIS RESULTS ---\n" + df_results.to_string(index=False))
            return df_results

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

def main() -> int:
    """Main execution function."""
    try:
        config = Config()
        processor = DataProcessor(config)
        results = processor.run_analysis()
        results.to_csv('performance_results.csv', index=False)
        logger.info("Results saved to performance_results.csv")
        return 0
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())