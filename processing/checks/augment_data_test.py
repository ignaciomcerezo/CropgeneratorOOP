from processing.augment_data.parallel.augment_data_parallel import augment_data_parallel

if __name__ == "__main__":
    augment_data_parallel(
        orders_to_consider=[1],
        generate_full_pages=True,
        max_samples_per_order=0,
        tasks_only=[5],
    )
