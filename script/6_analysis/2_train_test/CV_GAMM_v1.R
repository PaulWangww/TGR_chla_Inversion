pkgs <- c("dplyr", "tidyr", "mgcv", "ggplot2", "patchwork")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

set.seed(42)

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit <- grep("--file=", args, fixed = TRUE, value = TRUE)
  if (length(hit) > 0) return(dirname(normalizePath(sub("--file=", "", hit[1]), winslash = "/")))
  normalizePath(getwd(), winslash = "/")
}

script_dir <- get_script_dir()
workflow_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
data_path <- file.path(workflow_dir, "1_Data", "History", "History_data_chla_matched.csv")
out_dir <- file.path(script_dir, "output")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

driver_vars <- c(
  "Slope", "DL",
  "Tair", "P", "RH", "SW", "WS",
  "WL_mean", "WL_stability", "N_rapid_rise", "N_rapid_drop",
  "Aimp", "Acrop"
)
response_vars <- c("chla")
sc_suffix <- paste0(driver_vars, "_sc")
tnr <- "Times New Roman"

cat("========== 1. Load CHLA matched data ==========\n")
df <- read.csv(data_path, stringsAsFactors = FALSE) |>
  mutate(basin_id = as.factor(basin_id))

missing_cols <- setdiff(c("basin_id", "basin_name", "year", driver_vars, response_vars), names(df))
if (length(missing_cols) > 0) stop("Missing columns: ", paste(missing_cols, collapse = ", "))

cat("Rows:", nrow(df), "\n")
cat("Basins:", nlevels(df$basin_id), "\n")
cat("Years:", min(df$year), "-", max(df$year), "\n")

na_check <- sapply(c(driver_vars, response_vars), function(v) sum(is.na(df[[v]])))
if (any(na_check > 0)) {
  print(na_check[na_check > 0])
  stop("Missing values found in model inputs.")
}

sc_means <- sapply(driver_vars, function(v) mean(df[[v]], na.rm = TRUE))
sc_sds <- sapply(driver_vars, function(v) sd(df[[v]], na.rm = TRUE))

df_sc <- df |>
  mutate(across(
    all_of(driver_vars),
    ~ (. - sc_means[cur_column()]) / sc_sds[cur_column()],
    .names = "{.col}_sc"
  ))

safe_k <- function(var_name, k_target = 5) {
  n_uniq <- length(unique(df[[var_name]]))
  max(3L, min(k_target, n_uniq - 1L))
}

build_formula <- function(resp) {
  k_targets <- setNames(rep(5L, length(driver_vars)), driver_vars)
  terms <- mapply(function(sc_v, orig_v) {
    sprintf("s(%s, k=%d)", sc_v, safe_k(orig_v, k_targets[orig_v]))
  }, sc_suffix, driver_vars)
  rhs <- paste(c(terms, "s(basin_id, bs='re')"), collapse = " + ")
  as.formula(paste(resp, "~", rhs))
}

calc_metrics <- function(obs, pred, method, resp) {
  idx <- !is.na(obs) & !is.na(pred)
  o <- obs[idx]
  p <- pred[idx]
  ct <- cor.test(o, p)
  r <- round(unname(ct$estimate), 4)
  data.frame(
    method = method,
    response = resp,
    n = sum(idx),
    r = r,
    p_value = signif(ct$p.value, 3),
    R2 = round(r^2, 4),
    NSE = round(1 - sum((o - p)^2) / sum((o - mean(o))^2), 4),
    RMSE = round(sqrt(mean((o - p)^2)), 4)
  )
}

cat("\n========== 2. LOEO-CV ==========\n")
embayments <- levels(df_sc$basin_id)
loeo_preds <- setNames(lapply(response_vars, function(r) rep(NA_real_, nrow(df_sc))), response_vars)

for (i in seq_along(embayments)) {
  test_id <- embayments[i]
  train_idx <- df_sc$basin_id != test_id
  test_idx <- df_sc$basin_id == test_id
  cat(sprintf("  Fold %02d/%02d: basin_id=%s, n_test=%d\n", i, length(embayments), test_id, sum(test_idx)))

  for (resp in response_vars) {
    m <- gam(build_formula(resp), data = df_sc[train_idx, ], method = "REML",
             select = TRUE, family = gaussian())
    loeo_preds[[resp]][test_idx] <- predict(
      m, newdata = df_sc[test_idx, ],
      exclude = "s(basin_id)", newdata.guaranteed = TRUE
    )
  }
}

cat("\n========== 3. LOYO-CV ==========\n")
years <- sort(unique(df_sc$year))
loyo_preds <- setNames(lapply(response_vars, function(r) rep(NA_real_, nrow(df_sc))), response_vars)

for (yr in years) {
  train_idx <- df_sc$year != yr
  test_idx <- df_sc$year == yr
  train_data <- df_sc[train_idx, ]
  test_data <- df_sc[test_idx, ]
  cat(sprintf("  Year %d: n_test=%d\n", yr, sum(test_idx)))

  for (resp in response_vars) {
    m <- gam(build_formula(resp), data = train_data, method = "REML",
             select = TRUE, family = gaussian())
    pred_fixed <- predict(m, newdata = test_data,
                          exclude = "s(basin_id)", newdata.guaranteed = TRUE)
    terms_train <- predict(m, type = "terms")
    re_col_name <- grep("basin_id", colnames(terms_train), value = TRUE)
    if (length(re_col_name) == 1) {
      re_by_basin <- tapply(terms_train[, re_col_name], train_data$basin_id, mean)
      re_add <- re_by_basin[as.character(test_data$basin_id)]
      re_add[is.na(re_add)] <- 0
      loyo_preds[[resp]][test_idx] <- pred_fixed + re_add
    } else {
      loyo_preds[[resp]][test_idx] <- pred_fixed
    }
  }
}

cat("\n========== 4. CV metrics and plots ==========\n")
metrics_rows <- list()
for (resp in response_vars) {
  obs <- df_sc[[resp]]
  metrics_rows[[paste0("LOEO_", resp)]] <- calc_metrics(obs, loeo_preds[[resp]], "LOEO-CV", resp)
  metrics_rows[[paste0("LOYO_", resp)]] <- calc_metrics(obs, loyo_preds[[resp]], "LOYO-CV", resp)
}
metrics_df <- do.call(rbind, metrics_rows)
rownames(metrics_df) <- NULL
print(metrics_df)
write.csv(metrics_df, file.path(out_dir, "CV_metrics.csv"), row.names = FALSE)

make_scatter_plots <- function(preds_list, method_label) {
  plots <- lapply(response_vars, function(resp) {
    obs <- df_sc[[resp]]
    pred <- preds_list[[resp]]
    idx <- !is.na(obs) & !is.na(pred)
    dp <- data.frame(obs = obs[idx], pred = pred[idx])
    ct <- cor.test(dp$obs, dp$pred)
    r_val <- round(unname(ct$estimate), 3)
    p_lab <- if (ct$p.value < 0.001) "p < 0.001" else sprintf("p = %.3f", ct$p.value)
    rmse <- round(sqrt(mean((dp$obs - dp$pred)^2)), 3)

    ggplot(dp, aes(x = obs, y = pred)) +
      geom_point(alpha = 0.55, size = 2, color = "#1A5FA8") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
      geom_smooth(method = "lm", se = FALSE, color = "#FF6B35", linewidth = 0.8) +
      labs(title = paste(method_label, "-", resp),
           subtitle = sprintf("r = %.3f, %s, RMSE = %.3f", r_val, p_lab, rmse),
           x = "Observed CHLA", y = "Predicted CHLA") +
      theme_bw(base_size = 12, base_family = tnr)
  })
  wrap_plots(plots, ncol = 1)
}

ggsave(file.path(out_dir, "LOEO_CV_scatter.pdf"),
       make_scatter_plots(loeo_preds, "LOEO-CV"), width = 5.5, height = 5, device = cairo_pdf)
ggsave(file.path(out_dir, "LOYO_CV_scatter.pdf"),
       make_scatter_plots(loyo_preds, "LOYO-CV"), width = 5.5, height = 5, device = cairo_pdf)

p_bar <- ggplot(metrics_df, aes(x = response, y = r, fill = method)) +
  geom_col(position = position_dodge(width = 0.6), width = 0.5) +
  geom_text(aes(label = sprintf("%.3f", r)), position = position_dodge(width = 0.6),
            vjust = -0.4, size = 3.5, family = tnr) +
  scale_fill_manual(values = c("LOEO-CV" = "#1A5FA8", "LOYO-CV" = "#E8622A")) +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1))) +
  labs(title = "CHLA Cross-Validation Performance", x = "Response Variable",
       y = "Pearson r", fill = "CV Method") +
  theme_bw(base_size = 12, base_family = tnr) +
  theme(legend.position = "top")
ggsave(file.path(out_dir, "CV_r_comparison.pdf"), p_bar, width = 6, height = 5, device = cairo_pdf)

pred_detail <- df_sc[, c("basin_id", "basin_name", "year", response_vars)]
for (resp in response_vars) {
  pred_detail[[paste0(resp, "_LOEO_pred")]] <- loeo_preds[[resp]]
  pred_detail[[paste0(resp, "_LOYO_pred")]] <- loyo_preds[[resp]]
}
write.csv(pred_detail, file.path(out_dir, "CV_predictions_detail.csv"), row.names = FALSE)

cat("\n========== 5. Fit final full-data model ==========\n")
final_models <- list()
for (resp in response_vars) {
  m <- gam(build_formula(resp), data = df_sc, method = "REML", select = TRUE, family = gaussian())
  final_models[[resp]] <- m
  cat(sprintf("  %s: R-sq(adj)=%.4f, Dev.expl=%.1f%%\n",
              resp, summary(m)$r.sq, summary(m)$dev.expl * 100))
}

model_bundle <- list(
  models = final_models,
  sc_means = sc_means,
  sc_sds = sc_sds,
  driver_vars = driver_vars,
  sc_suffix = sc_suffix,
  response_vars = response_vars,
  basin_levels = levels(df_sc$basin_id),
  train_date = Sys.Date()
)

saveRDS(model_bundle, file.path(out_dir, "final_model_bundle_chla.rds"))
cat("Saved:", file.path(out_dir, "final_model_bundle_chla.rds"), "\n")
