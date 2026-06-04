pkgs <- c("dplyr", "mgcv", "ggplot2", "patchwork")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit <- grep("--file=", args, fixed = TRUE, value = TRUE)
  if (length(hit) > 0) return(dirname(normalizePath(sub("--file=", "", hit[1]), winslash = "/")))
  normalizePath(getwd(), winslash = "/")
}

script_dir <- get_script_dir()
workflow_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
bundle_path <- file.path(workflow_dir, "2_Train+Test", "output", "final_model_bundle_chla.rds")
data_path <- file.path(workflow_dir, "1_Data", "History", "History_data_chla_matched.csv")
out_dir <- file.path(script_dir, "output")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
tnr <- "Times New Roman"

cat("========== 1. Load CHLA model bundle ==========\n")
if (!file.exists(bundle_path)) stop("Model bundle not found: ", bundle_path)
bundle <- readRDS(bundle_path)

required_fields <- c("models", "sc_means", "sc_sds", "driver_vars", "sc_suffix", "response_vars", "basin_levels")
missing_fields <- setdiff(required_fields, names(bundle))
if (length(missing_fields) > 0) stop("Bundle missing fields: ", paste(missing_fields, collapse = ", "))
if (!identical(bundle$response_vars, "chla")) stop("Expected response_vars to be chla.")

cat("Train date:", as.character(bundle$train_date), "\n")
cat("Response:", paste(bundle$response_vars, collapse = ", "), "\n")

cat("\n========== 2. Load matched history and standardize ==========\n")
df_hist <- read.csv(data_path, stringsAsFactors = FALSE) |>
  mutate(basin_id = factor(basin_id, levels = bundle$basin_levels))

missing_cols <- setdiff(c(bundle$driver_vars, bundle$response_vars), names(df_hist))
if (length(missing_cols) > 0) stop("History data missing columns: ", paste(missing_cols, collapse = ", "))

for (v in bundle$driver_vars) {
  df_hist[[paste0(v, "_sc")]] <- (df_hist[[v]] - bundle$sc_means[v]) / bundle$sc_sds[v]
}

cat("Rows:", nrow(df_hist), "\n")
cat("Scaled NA count:", sum(is.na(df_hist[bundle$sc_suffix])), "\n")

cat("\n========== 3. Training-set back-prediction ==========\n")
pred_results <- list()
for (resp in bundle$response_vars) {
  pred_results[[resp]] <- predict(
    bundle$models[[resp]],
    newdata = df_hist,
    exclude = "s(basin_id)",
    newdata.guaranteed = TRUE
  )
  cat(resp, "NA predictions:", sum(is.na(pred_results[[resp]])), "\n")
}

calc_metrics <- function(obs, pred, resp) {
  idx <- !is.na(obs) & !is.na(pred)
  o <- obs[idx]
  p <- pred[idx]
  ct <- cor.test(o, p)
  r <- round(unname(ct$estimate), 4)
  data.frame(
    response = resp,
    n = sum(idx),
    r = r,
    p_value = signif(ct$p.value, 3),
    R2 = round(r^2, 4),
    NSE = round(1 - sum((o - p)^2) / sum((o - mean(o))^2), 4),
    RMSE = round(sqrt(mean((o - p)^2)), 4)
  )
}

metrics_df <- do.call(rbind, lapply(bundle$response_vars, function(resp) {
  calc_metrics(df_hist[[resp]], pred_results[[resp]], resp)
}))
rownames(metrics_df) <- NULL
print(metrics_df)
write.csv(metrics_df, file.path(out_dir, "test_metrics.csv"), row.names = FALSE)

plots <- lapply(bundle$response_vars, function(resp) {
  dp <- data.frame(obs = df_hist[[resp]], pred = pred_results[[resp]]) |>
    filter(is.finite(obs), is.finite(pred))
  ct <- cor.test(dp$obs, dp$pred)
  r_val <- round(unname(ct$estimate), 3)
  p_lab <- if (ct$p.value < 0.001) "p < 0.001" else sprintf("p = %.3f", ct$p.value)
  rmse <- round(sqrt(mean((dp$obs - dp$pred)^2)), 3)

  ggplot(dp, aes(x = obs, y = pred)) +
    geom_point(alpha = 0.55, size = 2, color = "#1A5FA8") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
    geom_smooth(method = "lm", se = FALSE, color = "#FF6B35", linewidth = 0.8) +
    labs(title = paste("Training Fit -", resp),
         subtitle = sprintf("r = %.3f, %s, RMSE = %.3f", r_val, p_lab, rmse),
         x = "Observed CHLA", y = "Predicted CHLA (excl. RE)") +
    theme_bw(base_size = 12, base_family = tnr)
})

ggsave(file.path(out_dir, "test_fit_scatter.pdf"),
       wrap_plots(plots, ncol = 1), width = 5.5, height = 5, device = cairo_pdf)

pred_df <- df_hist[, c("basin_id", "basin_name", "year", bundle$response_vars)]
for (resp in bundle$response_vars) pred_df[[paste0(resp, "_pred")]] <- pred_results[[resp]]
write.csv(pred_df, file.path(out_dir, "test_predictions.csv"), row.names = FALSE)

cat("Saved outputs in:", out_dir, "\n")
