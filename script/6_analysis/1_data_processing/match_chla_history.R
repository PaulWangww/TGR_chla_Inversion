pkgs <- c("readxl", "dplyr")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  hit <- grep(file_arg, args, fixed = TRUE, value = TRUE)
  if (length(hit) > 0) return(dirname(normalizePath(sub(file_arg, "", hit[1]), winslash = "/")))
  normalizePath(getwd(), winslash = "/")
}

script_dir <- get_script_dir()
workflow_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)

history_path <- file.path(workflow_dir, "1_Data", "History", "History data.xlsx")
chla_path <- file.path(workflow_dir, "1_Data", "History", "river_year_chla_median.csv")
out_path <- file.path(workflow_dir, "1_Data", "History", "History_data_chla_matched.csv")

cat("Reading history drivers:\n ", history_path, "\n")
hist <- readxl::read_excel(history_path, sheet = "Sheet1")

cat("Reading CHLA data:\n ", chla_path, "\n")
chla <- read.csv(chla_path, check.names = FALSE, stringsAsFactors = FALSE) |>
  dplyr::rename(
    basin_name = River_Folder,
    year = Year,
    chla = `chl-a_Median`
  )

matched <- hist |>
  dplyr::inner_join(chla, by = c("basin_name", "year")) |>
  dplyr::select(-dplyr::any_of(c("MBA", "MBP", "BO")), dplyr::everything()) |>
  dplyr::relocate(chla, .after = year)

unmatched <- chla |>
  dplyr::anti_join(hist, by = c("basin_name", "year"))

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write.csv(matched, out_path, row.names = FALSE)

cat("\nMatched rows:", nrow(matched), "\n")
cat("Matched basins:", dplyr::n_distinct(matched$basin_name), "\n")
cat("Matched years:", min(matched$year), "-", max(matched$year), "\n")
cat("CHLA missing values:", sum(is.na(matched$chla)), "\n")
cat("Unmatched CHLA rows:", nrow(unmatched), "\n")
cat("Saved:\n ", out_path, "\n")

if (nrow(matched) != 161) warning("Expected 161 matched rows, got ", nrow(matched), call. = FALSE)
if (dplyr::n_distinct(matched$basin_name) != 23) warning("Expected 23 basins.", call. = FALSE)
if (any(is.na(matched$chla))) warning("CHLA contains missing values.", call. = FALSE)
if (nrow(unmatched) > 0) warning("Some CHLA rows did not match history drivers.", call. = FALSE)
