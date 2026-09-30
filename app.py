from __future__ import annotations

import ast
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import scipy.signal as signal
import streamlit as st


st.set_page_config(
    page_title="آزمایشگاه پاسخ پله",
    page_icon=":material/monitoring:",
    layout="wide",
)


def parse_coefficients(value: str, label: str) -> np.ndarray:
    try:
        coefficients = np.array(
            [float(item) for item in value.replace(";", ",").split(",")],
            dtype=float,
        )
    except ValueError as error:
        raise ValueError(f"ضرایب {label} باید با کاما و به‌صورت عددی وارد شوند.") from error
    if coefficients.size == 0 or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"ضرایب {label} معتبر نیستند.")
    return coefficients


def parse_matrix(value: str, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(ast.literal_eval(value), dtype=float)
    except (SyntaxError, ValueError, TypeError) as error:
        raise ValueError(f"ماتریس {label} را با قالب [[...], [...]] وارد کنید.") from error
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"ماتریس {label} باید دوبعدی و شامل اعداد معتبر باشد.")
    return matrix


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=4, suppress_small=True)


def build_model(order: str, gain: float, parameter: float, damping: float) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    if order == "مرتبهٔ اول":
        time_constant = parameter
        numerator = np.array([gain])
        denominator = np.array([time_constant, 1.0])
    else:
        natural_frequency = parameter
        numerator = np.array([gain * natural_frequency**2])
        denominator = np.array([1.0, 2.0 * damping * natural_frequency, natural_frequency**2])
    state_space = signal.tf2ss(numerator, denominator)
    return numerator, denominator, state_space


def calculate_step(
    system: signal.TransferFunction | signal.StateSpace,
    duration: float,
    points: int,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.0, duration, points)
    response_time, response = signal.step(system, T=time)
    return response_time, amplitude * np.asarray(response, dtype=float)


st.title("آزمایشگاه پاسخ پله", icon=":material/monitoring:")
st.caption("مدل سیستم را تعریف کنید، نمایش معادل را ببینید و پاسخ زمانی را بررسی کنید.")

mode = st.segmented_control(
    "روش تعریف سیستم",
    ["تابع تبدیل", "فضای حالت", "سازندهٔ مدل"],
    default="سازندهٔ مدل",
    selection_mode="single",
)

if mode is None:
    st.stop()

with st.container(border=True):
    st.subheader("پارامترهای شبیه‌سازی", icon=":material/tune:")
    settings_left, settings_right, settings_time = st.columns(3)
    with settings_left:
        duration = st.number_input("مدت شبیه‌سازی (ثانیه)", min_value=0.1, value=10.0, step=1.0)
    with settings_right:
        points = st.number_input("تعداد نقاط", min_value=50, max_value=5000, value=500, step=50)
    with settings_time:
        amplitude = st.number_input("دامنهٔ پله", value=1.0, step=0.5)

system_data: tuple[str, Any] | None = None
model_description = ""

with st.form("system_form"):
    if mode == "تابع تبدیل":
        st.subheader("ورود تابع تبدیل")
        st.latex(r"G(s)=\frac{b_ms^m+\cdots+b_0}{a_ns^n+\cdots+a_0}")
        numerator_text = st.text_input("ضرایب صورت", value="1", help="از توان بالاتر به پایین‌تر، با کاما جدا کنید.")
        denominator_text = st.text_input("ضرایب مخرج", value="1, 2, 1", help="مثال: 1, 2, 1 یعنی s² + 2s + 1.")
    elif mode == "فضای حالت":
        st.subheader("ورود فضای حالت")
        st.latex(r"\dot{x}=Ax+Bu,\qquad y=Cx+Du")
        matrix_a = st.text_input("ماتریس A", value="[[-2, -1], [1, 0]]")
        matrix_b = st.text_input("ماتریس B", value="[[1], [0]]")
        matrix_c = st.text_input("ماتریس C", value="[[0, 1]]")
        matrix_d = st.text_input("ماتریس D", value="[[0]]")
        st.caption("ماتریس‌ها را به شکل فهرست پایتون وارد کنید؛ نمونه: [[1, 0], [0, 1]].")
    else:
        st.subheader("سازندهٔ مدل استاندارد")
        st.caption("با چند پارامتر ساده، تابع تبدیل و نمایش فضای حالت را هم‌زمان بسازید.")
        order = st.selectbox("مرتبهٔ سیستم", ["مرتبهٔ اول", "مرتبهٔ دوم"])
        builder_left, builder_mid, builder_right = st.columns(3)
        with builder_left:
            gain = st.number_input("بهرهٔ K", value=1.0, step=0.5)
        with builder_mid:
            if order == "مرتبهٔ اول":
                parameter = st.number_input("ثابت زمانی τ", min_value=0.01, value=1.0, step=0.25)
            else:
                parameter = st.number_input("فرکانس طبیعی ωₙ", min_value=0.01, value=2.0, step=0.25)
        with builder_right:
            damping = st.number_input("نسبت میرایی ζ", min_value=0.0, value=0.5, step=0.1, disabled=order == "مرتبهٔ اول")

    submitted = st.form_submit_button("محاسبه و رسم پاسخ", type="primary", icon=":material/play_arrow:")

if submitted:
    try:
        if mode == "تابع تبدیل":
            numerator = parse_coefficients(numerator_text, "صورت")
            denominator = parse_coefficients(denominator_text, "مخرج")
            numerator = np.trim_zeros(numerator, "f")
            denominator = np.trim_zeros(denominator, "f")
            if denominator.size == 0:
                raise ValueError("مخرج تابع تبدیل نمی‌تواند صفر باشد.")
            if numerator.size == 0:
                numerator = np.array([0.0])
            if numerator.size > denominator.size:
                raise ValueError("درجهٔ صورت نباید از درجهٔ مخرج بیشتر باشد.")
            system = signal.TransferFunction(numerator, denominator)
            state_space = signal.tf2ss(numerator, denominator)
            model_description = "تابع تبدیل واردشده"
        elif mode == "فضای حالت":
            matrix_a_value = parse_matrix(matrix_a, "A")
            matrix_b_value = parse_matrix(matrix_b, "B")
            matrix_c_value = parse_matrix(matrix_c, "C")
            matrix_d_value = parse_matrix(matrix_d, "D")
            state_count = matrix_a_value.shape[0]
            if matrix_a_value.shape != (state_count, state_count):
                raise ValueError("ماتریس A باید مربعی باشد.")
            if matrix_b_value.shape[0] != state_count or matrix_c_value.shape[1] != state_count:
                raise ValueError("ابعاد B و C باید با تعداد حالت‌های A سازگار باشند.")
            if matrix_b_value.shape[1] != 1 or matrix_c_value.shape[0] != 1:
                raise ValueError("در این نسخه، مدل فضای حالت باید یک ورودی و یک خروجی داشته باشد.")
            if matrix_d_value.shape != (matrix_c_value.shape[0], matrix_b_value.shape[1]):
                raise ValueError("ابعاد D باید با تعداد خروجی‌ها و ورودی‌های سیستم سازگار باشد.")
            state_space = (matrix_a_value, matrix_b_value, matrix_c_value, matrix_d_value)
            system = signal.StateSpace(*state_space)
            numerator, denominator = signal.ss2tf(*state_space)
            numerator = numerator[0]
            model_description = "فضای حالت واردشده"
        else:
            numerator, denominator, state_space = build_model(order, gain, parameter, damping)
            system = signal.TransferFunction(numerator, denominator)
            model_description = f"مدل استاندارد {order}"

        time, response = calculate_step(system, float(duration), int(points), float(amplitude))
        response_frame = pd.DataFrame({"زمان (ثانیه)": time, "خروجی": response})

        st.header("پاسخ سیستم")
        st.caption(model_description)
        chart = (
            alt.Chart(response_frame)
            .mark_line(color="#087f73", strokeWidth=3)
            .encode(
                x=alt.X("زمان (ثانیه):Q", title="زمان (ثانیه)"),
                y=alt.Y("خروجی:Q", title="خروجی", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("زمان (ثانیه):Q", format=".3f"), alt.Tooltip("خروجی:Q", format=".4f")],
            )
            .properties(height=390)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")

        final_value = float(response[-1])
        peak_value = float(np.max(response))
        peak_time = float(time[int(np.argmax(response))])
        metric_final, metric_peak, metric_time = st.columns(3)
        metric_final.metric("مقدار در انتهای بازه", f"{final_value:.4g}")
        metric_peak.metric("بیشینهٔ پاسخ", f"{peak_value:.4g}")
        metric_time.metric("زمان رسیدن به بیشینه", f"{peak_time:.4g} s")

        with st.expander("نمایش مدل ریاضی", icon=":material/functions:"):
            st.markdown("**تابع تبدیل**")
            st.code(f"صورت: {format_array(numerator)}\nمخرج: {format_array(denominator)}", language="text")
            st.markdown("**فضای حالت معادل**")
            state_labels = ("A", "B", "C", "D")
            state_text = "\n\n".join(f"{label} = {format_array(value)}" for label, value in zip(state_labels, state_space))
            st.code(state_text, language="text")
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        st.error(str(error), icon=":material/error:")
