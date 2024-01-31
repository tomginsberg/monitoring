import os
import sys
# Get the absolute path of the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Append the project root to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)

import pickle
from glob import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from scipy.stats import beta
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.utils import resample

from data.synthetic import SyntheticData
from detectron import detect
from training.train import process_dataset

# ----------------------------------
# Initialize the app

st.sidebar.image("app/logo-light.png", use_column_width='auto')

available_models = glob("models/*.model")
display_names = [os.path.basename(x).split(".")[0] for x in available_models]
model_option = st.sidebar.selectbox(
    'Choose a model:',
    display_names
)
with open(available_models[display_names.index(model_option)], "rb") as f:
    model_config = pickle.load(f)

model = model_config["model"]
model_config = model_config["information"]

# Placeholder for model information
st.sidebar.write("Information:")
st.sidebar.code(
    yaml.dump(model_config),
    language='yaml'
)

data = SyntheticData(n_samples=5000)
data = process_dataset(data, **model_config)
id_x_test, id_y_test, id_meta = data['id_test']['data'], data['id_test']['label'], data['id_test']['meta']
id_y_test = id_y_test.astype(int).to_numpy()
x_test, y_test, meta = data['od_test']['data'], data['od_test']['label'], data['od_test']['meta']
y_test = y_test.astype(int).to_numpy()
# split test data into two regimes pre-2021 and post 2021
mask = meta.date < pd.to_datetime("2022-01-01")
x_test_pre, y_test_pre, meta_pre = x_test[mask], y_test[mask], meta[mask]
x_test, y_test, meta = x_test[~mask], y_test[~mask], meta[~mask]

data_tab, eval_tab, shift_tab = st.tabs(["Data Explorer", "Historical Evaluation", "Realtime Monitoring"])


# ----------------------------------
# Evaluation functions


def get_color(health_value):
    return f"rgb({int(255 * (1 - health_value))}, {int(255 * health_value)}, 0)"


def bernoulli_confidence_interval(successes, trials):
    """
    Calculate the 95% confidence interval for the rate of a Bernoulli process
    using Bayesian inference with a uniform prior.

    Parameters:
    successes (int): Number of successes (True values).
    trials (int): Total number of trials.

    Returns:
    tuple: Lower and upper bounds of the 95% confidence interval.
    """
    # Parameters for the posterior Beta distribution
    alpha_prime = 1 + successes
    beta_prime = 1 + trials - successes

    # 95% confidence interval
    lower_bound = beta.ppf(0.025, alpha_prime, beta_prime)
    upper_bound = beta.ppf(0.975, alpha_prime, beta_prime)

    return lower_bound, upper_bound


def get_date_chunks_and_predictions(x, meta, n=30):
    first_day = meta.date.min()
    last_day = meta.date.max()
    date_chunks = pd.date_range(first_day, last_day, freq=f"{n}D")
    chunk_ids = meta.date.map(lambda x: np.argmin(np.abs(date_chunks - x)))
    predictions = model.predict_proba(x)
    return date_chunks, chunk_ids, predictions


def date_chunk_evaluator(x, y_test, meta, n=30, bootstrap_iterations=25, quantile=.95, eval_func=accuracy_score):
    date_chunks, chunk_ids, predictions = get_date_chunks_and_predictions(x, meta, n)
    predictions = predictions[:, 1]

    # TODO: should have a more general way of handling this
    if eval_func == accuracy_score:
        predictions = predictions > .5

    res_x = []
    res_y = []
    lower_bound = []
    upper_bound = []
    for chunk_id in sorted(chunk_ids.unique()):
        mask = chunk_ids == chunk_id
        y, p = y_test[mask], predictions[mask]

        res_x.append(date_chunks[chunk_id])

        # use Bayesian inference here
        if eval_func == accuracy_score:
            correct = np.sum(y == p)
            lower, upper = bernoulli_confidence_interval(correct, len(y))
            lower_bound.append(lower)
            upper_bound.append(upper)
        else:
            # resort to bootstrap
            scores = []
            for _ in range(bootstrap_iterations):
                indices = resample(range(len(y)))
                pred_score = eval_func(y[indices], p[indices])
                scores.append(pred_score)
            lower_bound.append(np.quantile(scores, 1 - quantile))
            upper_bound.append(np.quantile(scores, quantile))

        res_y.append(eval_func(y, p))

    return res_x, res_y, lower_bound, upper_bound


def atc_evaluator(x, meta, thresh, n=30, bootstrap_iterations=25, quantile=.95):
    date_chunks, chunk_ids, predictions = get_date_chunks_and_predictions(x, meta, n)

    res_x = []
    res_y = []
    lower_bound = []
    upper_bound = []
    confidences = predictions.max(1)
    for chunk_id in sorted(chunk_ids.unique()):
        mask = chunk_ids == chunk_id
        conf = confidences[mask]
        res_x.append(date_chunks[chunk_id])
        correct = np.sum(conf > thresh)
        lower, upper = bernoulli_confidence_interval(correct, len(conf))

        lower_bound.append(lower)
        upper_bound.append(upper)
        res_y.append(np.mean(conf > thresh))

    return res_x, res_y, lower_bound, upper_bound


# ----------------------------------
# Application tabs
with data_tab:
    features, labels, metadata = st.tabs(["Input Features", "Labels", "Metadata"])
    with features:

        fig = go.Figure()
        for col in x_test.columns:
            fig.add_trace(
                go.Violin(x=data['od_test']['data'][col], name=f'{col} (current)'))  # , histnorm='probability'))
            fig.add_trace(go.Violin(x=data['train']['data'][col], name=f'{col} (train)'))  # , histnorm='probability'))
        fig.update_layout(
            xaxis_title='Value',
            yaxis_title='Count',
            font=dict(
                size=14,
            )
        )
        st.plotly_chart(fig)
        # st.dataframe(x_test.describe().T)

    with labels:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=y_test, name='Label'))
        fig.update_layout(
            xaxis_title='Value',
            yaxis_title='Count',
            font=dict(
                size=14,
            )
        )
        st.plotly_chart(fig)
    with metadata:
        fig = go.Figure()
        for col in meta.columns:
            fig.add_trace(go.Histogram(x=meta[col], name=col))
        fig.update_layout(
            xaxis_title='Value',
            yaxis_title='Count',
            font=dict(
                size=14,
            )
        )
        st.plotly_chart(fig)
        # st.dataframe(meta.describe().T)

with eval_tab:
    n = st.slider("Evaluation Window (days)", 30, 100, 60, 10)
    res_x, res_y, lower_bound, upper_bound = date_chunk_evaluator(x_test_pre,
                                                                  y_test_pre, meta_pre, n, eval_func=accuracy_score)
    id_x, id_y, id_lower, id_upper = date_chunk_evaluator(id_x_test, id_y_test, id_meta, n, eval_func=accuracy_score)
    # prepend the end of id to the res
    res_x = [id_x[-1]] + res_x
    res_y = [id_y[-1]] + res_y
    lower_bound = [id_lower[-1]] + lower_bound
    upper_bound = [id_upper[-1]] + upper_bound

    fig = go.Figure(go.Scatter(x=res_x, y=res_y, mode='lines', name='After Training Cut-off', line=dict(dash='dash')))

    fig.add_trace(go.Scatter(
        x=res_x, y=lower_bound,
        line=dict(width=0),
        mode='lines',
        showlegend=False
    ))

    # Upper bound of the error with fill
    fig.add_trace(go.Scatter(
        x=res_x, y=upper_bound,
        fill='tonexty',  # Fill area between this trace and the next trace
        fillcolor='rgba(0,100,80,0.2)',  # Light transparent fill
        line=dict(width=0),
        mode='lines',
        name='95% Confidence Interval (After)',
    ))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Model Accuracy',
        font=dict(
            size=14,
        )
    )

    fig.add_trace(go.Scatter(
        x=id_x, y=id_y,
        mode='lines',
        name='Before Training Cut-off',
    ))

    fig.add_trace(go.Scatter(
        x=id_x, y=id_lower,
        line=dict(width=0),
        mode='lines',
        showlegend=False
    ))

    # Upper bound of the error with fill
    fig.add_trace(go.Scatter(
        x=id_x, y=id_upper,
        fill='tonexty',  # Fill area between this trace and the next trace
        fillcolor='rgba(0,100,80,0.2)',  # Light transparent fill
        line=dict(width=0),
        mode='lines',
        name='95% Confidence Interval (Before)',
    ))

    st.plotly_chart(fig)

with shift_tab:
    n = st.slider("Evaluation Window (days)", 30, 100, 60, 10, key=1)
    res_x, res_y, *_ = date_chunk_evaluator(x_test_pre, y_test_pre, meta_pre, n=n, eval_func=accuracy_score)
    thresh = model_config['atc_acc_thresh']
    atc_x_pre, atc_y_pre, lower_bound_pre, upper_bound_pre = atc_evaluator(x_test_pre, meta_pre, thresh=thresh, n=n)
    atc_x, atc_y, lower_bound, upper_bound = atc_evaluator(x_test, meta, thresh=thresh, n=n)

    fig = go.Figure()

    # Original accuracy line
    fig.add_trace(go.Scatter(x=res_x, y=res_y, mode='lines', name='Labeled Data'))
    # fig.add_trace(go.Scatter(x=atc_x, y=atc_y, mode='lines', name='Unlabeled Prediction', line=dict(dash='dash')))

    # Lower bound of the error (invisible trace)
    fig.add_trace(go.Scatter(
        x=atc_x_pre + atc_x, y=lower_bound_pre + lower_bound,
        line=dict(color='rgba(255,0,0,0.2)'),
        mode='lines',
        showlegend=False
    ))

    # Upper bound of the error with fill
    fig.add_trace(go.Scatter(
        x=atc_x_pre + atc_x, y=upper_bound_pre + upper_bound,
        fill='tonexty',  # Fill area between this trace and the next trace
        fillcolor='rgba(0,100,80,0.2)',  # Light transparent fill
        # line color red
        line=dict(color='rgba(255,0,0,0.2)'),
        # line=dict(width=0),
        mode='lines',
        name='Unlabeled Prediction (ACT)',
    ))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Accuracy',
        font=dict(size=14)
    )
    st.subheader("Realtime Performance Estimation")
    st.write("The model performance after deployment is estimated using the models"
             " confidence on unlabeled data."
             " The true performance is estimated using the most recently available labeled data"
             " that the model has not been trained on. For more details see the paper "
             " [Leveraging Unlabeled Data to Predict "
             "Out-of-Distribution Performance](https://arxiv.org/abs/2201.04234).")

    st.plotly_chart(fig)

    # make a scatter plot of the atc_y vs res_y, display the fit and r2 score

    fig = go.Figure()
    # pair res_y and atc_y based on res_x and atc_x
    fig.add_trace(go.Scatter(x=res_y, y=atc_y_pre, mode='markers', name='Calibration Data'))
    # add a line at y=x
    lower, upper = min(min(res_y), min(atc_y_pre)), max(max(res_y), max(atc_y_pre))
    # compute r2 score
    r2 = r2_score(res_y, atc_y_pre)
    fig.add_trace(go.Scatter(x=[lower, upper], y=[lower, upper], mode='lines', line=dict(dash='dash'),
                             name='r2={:.2f}'.format(r2)))
    fig.update_layout(
        xaxis_title='Model Accuracy',
        yaxis_title='Predicted Model Accuracy',
        font=dict(size=14)
    )

    st.subheader("Performance Estimator Calibration Curve")
    st.write("The calibration curve shows the relationship between the model's predicted accuracy"
             " and the true accuracy of the model. A well calibrated model will have a calibration curve"
             " that is close to the diagonal line.")
    st.plotly_chart(fig)

    f = lambda x: (x['data'], x['label'])

    st.subheader("Detectron Model Health")
    st.write("The model health is a measure of how well the model is performing "
             "based on the variations in it's output for a changing data stream."
             " See the [Detectron](https://arxiv.org/abs/2212.02742) paper for more details.")

    with st.spinner("Running **Detectron** (Calibration)"):
        sample_size = min(100, len(data['id_test']['data']))
        cal_record = detect.detectron_test_statistics(
            train=f(data['train']),
            val=f(data['val']),
            q=f(data['id_test']),
            base_model=model,
            sample_size=sample_size,
            balance_train_classes=False,
            ensemble_size=5,
            patience=3,
            verbose=False,
            test_weight_multiplier=10,
            show_progress_bar=False,
            # agree + test_weight_multiplier * alpha * disagree
        )
    with st.spinner("Running **Detectron** (Monitoring)"):
        results = []
        health = []
        dates = []

        for i in range(0, len(data['od_test']['data']) - sample_size, sample_size):
            dates.append(data['od_test']['meta'].iloc[[i, i + sample_size]]['date'])
            test_record = detect.detectron(
                train=f(data['train']),
                val=f(data['val']),
                observation_data=data['od_test']['data'].iloc[i:i + sample_size],
                calibration_record=cal_record,
                base_model=model,
                balance_train_classes=False,
                ensemble_size=5,
                patience=3,
                verbose=False,
                test_weight_multiplier=10,
                show_progress_bar=False,
            )
            results.append(test_record)
            health.append(min(1, results[-1].model_health))

        df = pd.DataFrame({"start_date": [(d.iloc[0]) for d in dates],
                           "end_date": [(d.iloc[1]) for d in dates],
                           "health": health})

        fig = go.Figure()
        for i in range(len(df)):
            fig.add_trace(
                go.Scatter(
                    x=[df.start_date.iloc[i], df.end_date.iloc[i]],
                    y=[df.health.iloc[i], df.health.iloc[i]],
                    fill='tozeroy',
                    fillcolor=get_color((df.health.iloc[i])),
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    showlegend=False,
                    mode='lines',
                    name='Health'
                )
            )

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Model Health',
            font=dict(size=14)
        )
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric(label="Realtime Model Health", value=f'{health[-1]:.2f}',
                delta=f'{health[-1] - health[-2]:.2f}')
    col2.metric(label='Latest Evaluation Date', value=f'{df.end_date.iloc[-1].strftime("%Y-%m-%d")}',
                delta=f'{(df.end_date.iloc[-1] - df.end_date.iloc[-2]).days} days')
    st.divider()
    st.plotly_chart(fig)
    # ----------------------------------
