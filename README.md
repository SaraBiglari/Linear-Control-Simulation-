# Linear-Control-Simulation-

# Step Response Lab

A Python application for defining control systems and plotting their step response with Streamlit.

## Running

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Ways to Define a System

- **Transfer function:** Enter the numerator and denominator coefficients in descending order of powers, separated by commas.
- **State space:** Enter the `A`, `B`, `C`, and `D` matrices using Python list format.
- **Model builder:** Build a first-order model `K/(τs + 1)` or a standard second-order model using gain, time constant/natural frequency, and damping ratio.

After defining the model, set the simulation duration, number of points, and step input amplitude, then select "Compute and Plot Response". The equivalent transfer function and state-space models are also shown in the "Mathematical Model" section.
