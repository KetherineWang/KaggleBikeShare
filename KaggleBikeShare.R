library(tidymodels)
library(lubridate)
library(tidyverse)
library(patchwork)
library(recipes)
library(ggplot2)
library(vroom)
library(dplyr)
library(glmnet)
library(poissonreg)
library(glue)

setwd("C:/Users/HongtingWang/Documents/STAT 348 - Predictive Analytics/KaggleBikeShare/data")

bike_train <- vroom("train.csv")
bike_test  <- vroom("test.csv")

glimpse(bike_train)

bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  update_role(datetime, new_role = "id") %>%
  step_mutate(
    datetime = ymd_hms(datetime),
    year = year(datetime),
    month = month(datetime),
    day = day(datetime),
    hour = hour(datetime),
    wday = wday(datetime, label = TRUE),
    week = week(datetime),
    quarter = quarter(datetime)
  ) %>%
  step_rm(casual, registered) %>%
  step_mutate(
    season = as.factor(season),
    holiday = as.factor(holiday),
    workingday = as.factor(workingday),
    weather = as.factor(weather),
    year = as.factor(year),
    month = as.factor(month),
    hour = as.factor(hour),
    wday = as.factor(wday),
    quarter = as.factor(quarter)
  ) %>%
  step_log(all_outcomes(), offset = 1) %>%
  step_log(windspeed, offset = 1) %>%
  step_normalize(temp, atemp, humidity, windspeed) %>%
  step_interact(terms = ~ hour:workingday) %>%
  step_interact(terms = ~ temp:humidity) %>%
# step_poly(hour, degree = 2) %>%
  step_dummy(all_nominal_predictors())

prepped <- prep(bike_recipe)
baked_train <- bake(prepped, new_data = NULL)

glimpse(baked_train)
# ---- Prepare train/test (NO log transform; Poisson model expects counts) ----
train_data <- bike_train %>%
  select(-casual, -registered)

test_data  <- bike_test

# ---- Recipe: no categorical variables left after encoding + normalization ----
# - Collapse weather==4 into 3 (as you had)
# - Extract hour from datetime
# - Remove datetime (no raw timestamps in model)
# - Dummy-code ALL nominal predictors
# - Normalize all numeric predictors
bike_recipe <- recipe(count ~ ., data = train_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(hour = lubridate::hour(datetime)) %>%
  step_rm(datetime) %>%
  # convert integer-coded categoricals to factors so they get dummy-encoded
  step_mutate(season = factor(season),
              weather = factor(weather),
              holiday = factor(holiday),
              workingday = factor(workingday)) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_normalize(all_numeric_predictors())

# Prep/bake just to show the first 5 rows per your Step 5 requirement
bike_prep   <- prep(bike_recipe, training = train_data)
train_baked <- bake(bike_prep, new_data = train_data)
test_baked  <- bake(bike_prep, new_data = test_data)

cat("\n=== First 5 rows: TRAIN (baked) ===\n")
print(utils::head(train_baked, 5))

cat("\n=== First 5 rows: TEST (baked) ===\n")
print(utils::head(test_baked, 5))

# ---- Penalized regression (Poisson via glmnet) ----
# Mixture \in [0,1] (0=ridge, 1=lasso); Penalty > 0
mixtures <- c(0.25, 0.50, 0.75, 1.00)
penalties <- c(1.0, 2.0, 3.0)

# Create all combinations (at least 5 total; this gives 25)
grid <- expand_grid(mixture = mixtures, penalty = penalties)

# Workflow skeleton
preg_wf <- workflow() %>%
  add_recipe(bike_recipe)

# Directory to write CSVs
out_dir <- getwd()

# Track output CSV filenames
out_files <- c()

for (i in seq_len(nrow(grid))) {
  mix <- grid$mixture[i]
  pen <- grid$penalty[i]
  
  # Define Poisson glmnet model with specific penalty/mixture
  preg_model <- poisson_reg(penalty = pen, mixture = mix) %>%
    set_engine("glmnet")
  
  wf <- preg_wf %>% add_model(preg_model)
  
  # Fit
  fit_obj <- fit(wf, data = train_data)
  
  # Predict on test set
  preds <- predict(fit_obj, new_data = test_data, type = "numeric") %>%
    transmute(count = pmax(0, .pred))  # clamp to >= 0
  
  # Build Kaggle submission
  kaggle_submission <- test_data %>%
    select(datetime) %>%
    mutate(datetime = as.character(format(datetime))) %>%
    bind_cols(preds)
  
  # Write a distinct file per (penalty, mixture)
  file_name <- glue("Kaggle_Poisson_pen{format(pen, scientific = FALSE)}_mix{mix}.csv")
  vroom_write(kaggle_submission, file = file.path(out_dir, file_name), delim = ",")
  
  out_files <- c(out_files, file_name)
}

cat("\nWrote the following Kaggle CSVs:\n")
print(out_files)



# Set working directory to the project dataset folder
setwd("C:/Users/HongtingWang/Documents/STAT 348 - Predictive Analytics/KaggleBikeShare/bike-sharing-demand-dataset")

bike_train <- vroom("train.csv")
bike_test <- vroom("test.csv")

head(bike_train)
glimpse(bike_test)

train_data <- bike_train %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

test_data  <- bike_test

bike_recipe <- recipe(count ~ ., data = train_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_num2factor(season, levels = c("1","2","3","4")) %>%
  step_num2factor(weather, levels = c("1","2","3")) %>%
  step_mutate(hour = lubridate::hour(datetime)) %>%
  step_normalize(all_numeric_predictors())

lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model)

bike_fit <- fit(bike_workflow, data = train_data)

preds_log <- predict(bike_fit, new_data = test_data)

bike_predictions <- preds_log %>%
  mutate(count = exp(.pred)) %>%
  mutate(count = pmax(0, count))

bike_prep   <- prep(bike_recipe, training = train_data)
train_baked <- bake(bike_prep, new_data = train_data)
test_baked  <- bake(bike_prep, new_data = test_data)

cat("\n=== First 5 rows: TRAIN (baked) ===\n")
print(utils::head(train_baked, 5))

cat("\n=== First 5 rows: TEST (baked) ===\n")
print(utils::head(test_baked, 5))

kaggle_submission <- test_data %>%
  select(datetime) %>%
  mutate(datetime = as.character(format(datetime))) %>%
  bind_cols(bike_predictions %>% select(count))

# Write to CSV
vroom_write(kaggle_submission, file = "./WorkflowPreds.csv", delim = ",")



# Type Conversion
train_data <- bike_train %>%
  mutate(
    season     = factor(season,     levels = 1:4, labels = c("Spring","Summer","Fall","Winter")),
    holiday    = factor(holiday,    levels = c(0,1), labels = c("No","Yes")),
    workingday = factor(workingday, levels = c(0,1), labels = c("No","Yes")),
    weather    = factor(weather,    levels = 1:4, labels = c("Clear","Mist/Cloudy","Light Snow/Rain","Heavy Rain/Snow"))
  ) %>%
  select(-any_of(c("casual", "registered")))

# Test preprocessing (match levels to train)
test_data <- bike_test %>%
  mutate(
    season     = factor(season,     levels = 1:4, labels = c("Spring","Summer","Fall","Winter")),
    holiday    = factor(holiday,    levels = c(0,1), labels = c("No","Yes")),
    workingday = factor(workingday, levels = c(0,1), labels = c("No","Yes")),
    weather    = factor(weather,    levels = 1:4, labels = c("Clear","Mist/Cloudy","Light Snow/Rain","Heavy Rain/Snow"))
  )

# Check dataset structure
glimpse(train_data)
glimpse(test_data)

# Linear Regression
my_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  fit(count ~ ., data = train_data)

# Predict
bike_predictions <- predict(my_linear_model, new_data = test_data)
bike_predictions

# Log
# Fit on log(1 + count); still drop casual & registered
my_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  fit(log1p(count) ~ ., data = train_data)

# Predict and convert back to the original scale
log_preds <- predict(my_linear_model, new_data = test_data)  # .pred is on log scale
bike_predictions <- log_preds %>%
  dplyr::transmute(.pred_count = expm1(.pred))

bike_predictions

kaggle_submission <- bike_predictions %>%
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>%
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")

# Check for missing values
colSums(is.na(bike_train))

# Summary statistics
summary(bike_train)

bike_train %>%
  select(temp, atemp, humidity, windspeed, count) %>%
  cor()

# Panel 1: Average count by weather (barplot required)
p1 <- ggplot(train, aes(x = weather, y = count, fill = weather)) +
  geom_bar(stat = "sum") +
  labs(title = "Average Bike Rentals by Weather", x = "Weather", y = "Avg Count") +
  theme_minimal()

# Panel 2: Average rentals by season
p2 <- ggplot(train, aes(x = season, y = count, fill = season)) +
  geom_bar(stat = "sum") +
  labs(title = "Average Bike Rentals by Season", x = "Season", y = "Avg Count") +
  theme_minimal()

# Panel 3: Rentals by hour of the day
p3 <- ggplot(train, aes(x = factor(hour), y = count, fill = factor(hour))) +
  geom_bar(stat = "sum") +
  labs(title = "Average Rentals by Hour", x = "Hour", y = "Avg Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Panel 4: Rentals on Working Days vs Holidays
p4 <- ggplot(train, aes(x = workingday, y = count, fill = workingday)) +
  geom_bar(stat = "sum") +
  labs(title = "Average Rentals: Working Day vs Holiday", x = "Working Day", y = "Avg Count") +
  theme_minimal()

(p1 | p2) / (p3 | p4)

ggsave("HW - Quant EDA.png", plot = (p1 | p2) / (p3 | p4), width = 12, height = 8, dpi = 300)

# Linear Regression

train_data <- bike_train %>%
  mutate(
    season = factor(season,
                    labels = c("Spring", "Summer", "Fall", "Winter")),
    holiday = factor(holiday,
                     labels = c("No", "Yes")),
    workingday = factor(workingday,
                        labels = c("No", "Yes")),
    weather = factor(weather,
                     labels = c("Clear", "Mist/Cloudy", "Light Snow/Rain", "Heavy Rain/Snow"))
  )

glimpse(train_data)

my_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  fit(count ~ . - casual - registered, data = train_data)

bike_predictions <- predict(my_linear_model, new_data=bike_test)

bike_predictions

kaggle_submission <- bike_predictions %>%
bind_cols(., testData) %>% 
  select(datetime, .pred) %>%
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")
