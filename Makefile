OUT_DIR ?= ./barcode_sr_data
NUM_SAMPLES ?= 1000
HR_SIZE ?= 512
SEED ?= 42
BARCODE_TYPE ?= 1d
CONFIG_OUT ?= $(OUT_DIR)/train_config.yml
# Set this to path of your .pth file if fine-tuning
PRETRAINED ?= 

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  generate           : Generate dataset (default 1d)"
	@echo "  generate_1d        : Generate 1D barcodes only"
	@echo "  generate_2d        : Generate 2D barcodes only"
	@echo "  generate_all       : Generate both 1D and 2D barcodes"
	@echo "  generate_fine_tune : Generate dataset AND training config (requires optional PRETRAINED=/path/to/weights.pth)"
	@echo "  train              : Start training using config in OUT_DIR"
	@echo ""
	@echo "Variables:"
	@echo "  OUT_DIR            : Output directory (default: ./barcode_sr_data)"
	@echo "  NUM_SAMPLES        : Number of samples (default: 1000)"
	@echo "  HR_SIZE            : High-res image size (default: 512)"
	@echo "  BARCODE_TYPE       : 1d, 2d, or all"

.PHONY: generate
generate:
	uv run python generate_barcode_sr_dataset.py \
		--out_dir $(OUT_DIR) \
		--num_samples $(NUM_SAMPLES) \
		--hr_size $(HR_SIZE) \
		--seed $(SEED) \
		--barcode_type $(BARCODE_TYPE)

.PHONY: generate_1d
generate_1d:
	$(MAKE) generate BARCODE_TYPE=1d OUT_DIR=./barcode_sr_data_1d

.PHONY: generate_2d
generate_2d:
	$(MAKE) generate BARCODE_TYPE=2d OUT_DIR=./barcode_sr_data_2d

.PHONY: generate_all
generate_all:
	$(MAKE) generate BARCODE_TYPE=all OUT_DIR=./barcode_sr_data_all

.PHONY: generate_fine_tune
generate_fine_tune:
	uv run python generate_barcode_sr_dataset.py \
		--out_dir $(OUT_DIR) \
		--num_samples $(NUM_SAMPLES) \
		--hr_size $(HR_SIZE) \
		--seed $(SEED) \
		--barcode_type $(BARCODE_TYPE) \
		--generate_config \
		--config_out $(CONFIG_OUT) \
		$(if $(PRETRAINED),--pretrained_weights $(PRETRAINED),)

.PHONY: train
train:
	uv run python start_fine_tuning.py --dataset_dir $(OUT_DIR)


