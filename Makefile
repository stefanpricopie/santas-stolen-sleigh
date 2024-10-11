# Define the competition name as a variable
COMPETITION_NAME = santas-stolen-sleigh-old
ZIP_FILE = $(COMPETITION_NAME).zip
DATA_DIR = data

# Default target (if you run `make` without arguments)
all: download_data unzip_data

# Target to download the competition data using Kaggle API
download_data:
	@echo "Downloading data from Kaggle competition: $(COMPETITION_NAME)"
	kaggle competitions download -c $(COMPETITION_NAME)

# Target to unzip the downloaded competition data
unzip_data: $(ZIP_FILE)
	@echo "Unzipping the data to the $(DATA_DIR) directory"
	unzip -o $(ZIP_FILE) -d $(DATA_DIR)

# Clean target to remove the downloaded zip file and the data directory
clean:
	rm -f $(ZIP_FILE)
	rm -rf $(DATA_DIR)

# Mark the ZIP_FILE as a file so that it doesn't get interpreted as a command
$(ZIP_FILE): download_data