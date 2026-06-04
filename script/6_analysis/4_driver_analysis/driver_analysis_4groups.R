pkgs <- c("dplyr", "mgcv", "ggplot2", "patchwork", "scales")
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
smooth_out_dir <- file.path(script_dir, "output_5x3")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(smooth_out_dir, recursive = TRUE, showWarnings = FALSE)

n_perm <- 500
tnr <- "Times New Roman"
driver_vars <- c(
  "Slope", "DL",
  "Tair", "P", "RH", "SW", "WS",
  "WL_mean", "WL_stability", "N_rapid_rise", "N_rapid_drop",
  "Aimp", "Acrop"
)
response_vars <- c("chla")
sc_suffix <- paste0(driver_vars, "_sc")

save_plot <- function(plot, base_path, width, height) {
  ggsave(paste0(base_path, ".pdf"), plot, width = width, height = height, device = cairo_pdf)
  ggsave(paste0(base_path, ".png"), plot, width = width, height = height, dpi = 320,
         device = grDevices::png, type = "cairo")
}

cat("========== 1. Load CHLA matched data ==========\n")
df <- read.csv(data_path, stringsAsFactors = FALSE) |>
  mutate(basin_id = as.factor(basin_id))

missing_cols <- setdiff(c("basin_id", "year", driver_vars, response_vars), names(df))
if (length(missing_cols) > 0) stop("Missing columns: ", paste(missing_cols, collapse = ", "))

na_check <- sapply(c(driver_vars, response_vars), function(v) sum(is.na(df[[v]])))
if (any(na_check > 0)) stop("Missing values found: ", paste(names(na_check)[na_check > 0], collapse = ", "))

sc_means <- sapply(driver_vars, function(v) mean(df[[v]], na.rm = TRUE))
sc_sds <- sapply(driver_vars, function(v) sd(df[[v]], na.rm = TRUE))
df_sc <- df |>
  mutate(across(all_of(driver_vars),
                ~ (. - sc_means[cur_column()]) / sc_sds[cur_column()],
                .names = "{.col}_sc"))

safe_k <- function(var_name, k_target = 5) {
  n_uniq <- length(unique(df[[var_name]]))
  max(3L, min(k_target, n_uniq - 1L))
}

build_formula <- function(resp) {
  k_targets <- setNames(rep(5L, length(driver_vars)), driver_vars)
  terms <- mapply(function(sc_v, orig_v) {
    sprintf("s(%s, k=%d)", sc_v, safe_k(orig_v, k_targets[orig_v]))
  }, sc_suffix, driver_vars)
  as.formula(paste(resp, "~", paste(c(terms, "s(basin_id, bs='re')"), collapse = " + ")))
}

cat("========== 2. Fit CHLA GAMM ==========\n")
models <- list()
for (resp in response_vars) {
  m <- gam(build_formula(resp), data = df_sc, method = "REML", select = TRUE, family = gaussian())
  models[[resp]] <- m
  cat(sprintf("%s: R-sq(adj)=%.3f, Dev.expl=%.1f%%\n",
              resp, summary(m)$r.sq, summary(m)$dev.expl * 100))
  sink(file.path(out_dir, paste0("gamm_summary_", resp, ".txt")))
  cat("Response variable:", resp, "\n\n")
  print(summary(m))
  sink()
}

build_smooth_plots <- function(resp) {
  m <- models[[resp]]
  sm_table <- summary(m)$s.table
  intercept <- coef(m)["(Intercept)"]
  template <- df_sc[1, , drop = FALSE]
  for (sv in sc_suffix) template[[sv]] <- 0
  template$basin_id <- levels(df_sc$basin_id)[1]

  plot_list <- list()
  for (i in seq_along(driver_vars)) {
    orig_var <- driver_vars[i]
    sc_var <- sc_suffix[i]
    x_orig <- seq(min(df[[orig_var]], na.rm = TRUE), max(df[[orig_var]], na.rm = TRUE), length.out = 100)
    x_sc <- (x_orig - sc_means[orig_var]) / sc_sds[orig_var]
    grid <- template[rep(1, 100), ]
    rownames(grid) <- NULL
    grid[[sc_var]] <- x_sc
    pred <- predict(m, newdata = grid, se.fit = TRUE, exclude = "s(basin_id)", newdata.guaranteed = TRUE)
    fit_c <- pred$fit - intercept
    plot_df <- data.frame(x = x_orig, fit = fit_c,
                          lower = fit_c - 1.96 * pred$se.fit,
                          upper = fit_c + 1.96 * pred$se.fit)
    sm_row <- sm_table[grepl(sc_var, rownames(sm_table), fixed = TRUE), , drop = FALSE]
    if (nrow(sm_row) > 0) {
      pv <- sm_row[1, "p-value"]
      edf <- sm_row[1, "edf"]
      p_lab <- sprintf("edf=%.2f  p=%s", edf, ifelse(pv < 0.001, "<0.001", round(pv, 3)))
      is_sig <- pv < 0.05
    } else {
      p_lab <- ""
      is_sig <- FALSE
    }
    plot_list[[orig_var]] <- ggplot(plot_df, aes(x = x, y = fit)) +
      geom_ribbon(aes(ymin = lower, ymax = upper), fill = if (is_sig) "#4E9AF1" else "grey80", alpha = 0.3) +
      geom_line(color = if (is_sig) "#1A5FA8" else "grey60", linewidth = 1) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.4) +
      labs(title = orig_var, subtitle = p_lab, x = orig_var, y = "Partial Effect") +
      theme_bw(base_size = 18, base_family = tnr)
  }
  padded <- c(plot_list[1:12], list(plot_spacer()), plot_list[13], list(plot_spacer()))
  wrap_plots(padded, ncol = 3)
}

for (resp in response_vars) {
  save_plot(build_smooth_plots(resp), file.path(smooth_out_dir, paste0("smooth_curves_", resp)),
            width = 13.5, height = 18)
}

groups <- list(
  physical = paste0(c("Slope", "DL"), "_sc"),
  climate = paste0(c("Tair", "P", "RH", "SW", "WS"), "_sc"),
  hydrodynamics = paste0(c("WL_mean", "WL_stability", "N_rapid_rise", "N_rapid_drop"), "_sc"),
  human = paste0(c("Aimp", "Acrop"), "_sc")
)

permute_group_mse <- function(model, group_vars, n_repeats = n_perm) {
  resp_col <- all.vars(formula(model))[1]
  y <- model$model[[resp_col]]
  f_orig <- fitted(model)
  terms_mat <- predict(model, type = "terms")
  grp_cols <- paste0("s(", group_vars, ")")
  grp_cols <- grp_cols[grp_cols %in% colnames(terms_mat)]
  baseline_mse <- mean((y - f_orig)^2, na.rm = TRUE)
  mses <- numeric(n_repeats)
  for (i in seq_len(n_repeats)) {
    f_perm <- f_orig
    for (col in grp_cols) {
      perm_idx <- sample.int(length(f_orig))
      f_perm <- f_perm - terms_mat[, col] + terms_mat[perm_idx, col]
    }
    mses[i] <- mean((y - f_perm)^2, na.rm = TRUE)
  }
  mean(mses, na.rm = TRUE) - baseline_mse
}

cat("========== 3. PRDA and variable importance ==========\n")
group_colors <- c(physical = "#4CAF50", climate = "#2196F3", hydrodynamics = "#FF9800", human = "#F44336")
group_labels <- c(physical = "Watershed Attributes", climate = "Climate",
                  hydrodynamics = "Hydrodynamics", human = "Human Activities")
var_group_map <- c(
  setNames(rep("physical", 2), c("Slope", "DL")),
  setNames(rep("climate", 5), c("Tair", "P", "RH", "SW", "WS")),
  setNames(rep("hydrodynamics", 4), c("WL_mean", "WL_stability", "N_rapid_rise", "N_rapid_drop")),
  setNames(rep("human", 2), c("Aimp", "Acrop"))
)

prda_results <- data.frame()
importance_list <- list()
for (resp in response_vars) {
  m <- models[[resp]]
  contribs <- pmax(sapply(groups, function(gv) permute_group_mse(m, gv)), 0)
  pct <- contribs / sum(contribs) * 100
  prda_results <- rbind(prda_results, data.frame(response = resp, group = names(pct),
                                                 delta_mse = round(contribs, 6),
                                                 contribution = round(pct, 2)))
  imp <- pmax(sapply(sc_suffix, function(sc_v) permute_group_mse(m, sc_v)), 0)
  names(imp) <- driver_vars
  pct_imp <- imp / sum(imp) * 100
  importance_list[[resp]] <- data.frame(variable = driver_vars, sc_var = sc_suffix,
                                        group = var_group_map[driver_vars],
                                        delta_mse = round(imp, 6),
                                        importance = round(pct_imp, 2),
                                        response = resp,
                                        stringsAsFactors = FALSE)
}

write.csv(prda_results, file.path(out_dir, "PRDA_contributions.csv"), row.names = FALSE)
importance_df <- do.call(rbind, importance_list)
rownames(importance_df) <- NULL
write.csv(importance_df, file.path(out_dir, "variable_importance.csv"), row.names = FALSE)

prda_results$group <- factor(prda_results$group, levels = names(group_colors))
p_prda <- ggplot(prda_results, aes(x = response, y = contribution, fill = group)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = ifelse(contribution > 3, paste0(round(contribution, 1), "%"), "")),
            position = position_stack(vjust = 0.5), color = "white", size = 3.5, fontface = "bold", family = tnr) +
  scale_fill_manual(values = group_colors, labels = group_labels) +
  scale_y_continuous(labels = percent_format(scale = 1), limits = c(0, 105), expand = expansion(mult = c(0, 0))) +
  labs(title = "Relative Contribution of Driving Factor Groups (CHLA)",
       subtitle = paste0("n_perm = ", n_perm), x = "Response Variable",
       y = "Relative Contribution (%)", fill = "Factor Group") +
  theme_bw(base_size = 12, base_family = tnr)
save_plot(p_prda, file.path(out_dir, "PRDA_barplot"), width = 7, height = 5)

group_imp_df <- importance_df |>
  group_by(response, group) |>
  summarise(contribution = sum(importance), .groups = "drop") |>
  mutate(group = factor(group, levels = names(group_colors)))
p_group_imp <- ggplot(group_imp_df, aes(x = response, y = contribution, fill = group)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = ifelse(contribution > 3, paste0(round(contribution, 1), "%"), "")),
            position = position_stack(vjust = 0.5), color = "white", size = 3.5, fontface = "bold", family = tnr) +
  scale_fill_manual(values = group_colors, labels = group_labels) +
  scale_y_continuous(labels = percent_format(scale = 1), limits = c(0, 105), expand = expansion(mult = c(0, 0))) +
  labs(title = "Relative Importance of Driving Factor Groups (CHLA)",
       subtitle = paste0("n_perm = ", n_perm), x = "Response Variable",
       y = "Relative Importance (%)", fill = "Factor Group") +
  theme_bw(base_size = 12, base_family = tnr)
save_plot(p_group_imp, file.path(out_dir, "variable_importance_barplot"), width = 7, height = 5)

for (resp in response_vars) {
  df_imp <- importance_list[[resp]] |>
    arrange(importance) |>
    mutate(variable = factor(variable, levels = variable))
  p_imp <- ggplot(df_imp, aes(x = importance, y = variable, fill = group)) +
    geom_col(width = 0.65) +
    geom_text(aes(label = sprintf("%.1f%%", importance)), hjust = -0.15, size = 3.2, family = tnr) +
    scale_fill_manual(values = group_colors, labels = group_labels) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15)), labels = function(x) paste0(x, "%")) +
    labs(title = paste("Variable Importance -", resp), subtitle = paste0("n_perm = ", n_perm),
         x = "Relative Importance (%)", y = NULL, fill = "Factor Group") +
    theme_bw(base_size = 11, base_family = tnr)
  save_plot(p_imp, file.path(out_dir, paste0("variable_importance_", resp)), width = 7, height = 6)
  save_plot(p_imp, file.path(out_dir, "variable_importance"), width = 7, height = 6)
}

cat("Done. Outputs in:", out_dir, "\n")
