import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="E-Commerce Analytics Platform",
    page_icon="🛒",
    layout="wide"
)
st.title("E-Commerce Customer Analytics Platform")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_excel("Online Retail.xlsx")
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df

with st.spinner("Loading data..."):
    df = load_data()

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Customer Segments", "Churn Prediction", "Sentiment Analysis"]
)

if page == "Overview":
    st.subheader("Business Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"£{df['Revenue'].sum():,.0f}")
    col2.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
    col3.metric("Total Transactions", f"{df['InvoiceNo'].nunique():,}")
    col4.metric("Total Products", f"{df['StockCode'].nunique():,}")
    st.markdown("---")

    revenue_time = df.groupby(
        df["InvoiceDate"].dt.to_period("M")
    )["Revenue"].sum().reset_index()
    revenue_time["InvoiceDate"] = revenue_time["InvoiceDate"].astype(str)

    fig = px.line(
        revenue_time,
        x="InvoiceDate",
        y="Revenue",
        title="Monthly Revenue Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top_countries = (
            df.groupby("Country")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig2 = px.bar(
            top_countries,
            x="Country",
            y="Revenue",
            title="Top 10 Countries by Revenue",
            color="Revenue",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        top_products = (
            df.groupby("Description")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig3 = px.bar(
            top_products,
            x="Revenue",
            y="Description",
            orientation="h",
            title="Top 10 Products by Revenue",
            color="Revenue",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig3, use_container_width=True)

elif page == "Customer Segments":
    st.subheader("Customer Segmentation")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby("Cluster").agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        Count=("CustomerID", "count")
    ).round(2)

    labels = {}
    for cluster in cluster_summary.index:
        r = cluster_summary.loc[cluster, "Recency"]
        f = cluster_summary.loc[cluster, "Frequency"]
        if r < 50 and f > 5:
            labels[cluster] = "Champions"
        elif r < 100 and f > 3:
            labels[cluster] = "Loyal Customers"
        elif r > 200:
            labels[cluster] = "Lost Customers"
        else:
            labels[cluster] = "At Risk"

    rfm["Segment"] = rfm["Cluster"].map(labels)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            rfm,
            names="Segment",
            title="Customer Segments Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.scatter_3d(
            rfm,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color="Segment",
            title="RFM 3D Segmentation",
            opacity=0.7
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Segment Summary")
    st.dataframe(cluster_summary, use_container_width=True)

elif page == "Churn Prediction":
    st.subheader("Churn Prediction Model")
    rfm["Churn"] = (rfm["Recency"] > 90).astype(int)
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features]
    y = rfm["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    rfm["Churn_Probability"] = xgb.predict_proba(X_scaled)[:, 1]
    churn_rate = rfm["Churn"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{accuracy:.2%}")
    col2.metric("Overall Churn Rate", f"{churn_rate:.2%}")
    col3.metric(
        "At Risk Customers",
        f"{rfm[rfm['Churn_Probability'] > 0.7].shape[0]:,}"
    )

    col1, col2 = st.columns(2)
    with col1:
        importance = pd.Series(
            xgb.feature_importances_,
            index=features
        ).sort_values(ascending=True).reset_index()
        importance.columns = ["Feature", "Importance"]
        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            color="Importance",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(
            rfm,
            x="Churn_Probability",
            color="Churn",
            title="Churn Probability Distribution",
            barmode="overlay",
            color_discrete_map={0: "green", 1: "red"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Top 10 Customers Most At Risk")
    at_risk = rfm.sort_values(
        "Churn_Probability", ascending=False
    ).head(10)
    st.dataframe(
        at_risk[[
            "CustomerID", "Recency",
            "Frequency", "Monetary",
            "Churn_Probability"
        ]].round(2),
        use_container_width=True
    )

elif page == "Sentiment Analysis":
    st.subheader("Product Sentiment Analysis")
    df_products = (
        df[["StockCode", "Description", "Revenue"]]
        .dropna()
        .drop_duplicates(subset="StockCode")
    )

    def get_sentiment(text):
        polarity = TextBlob(str(text)).sentiment.polarity
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"

    def get_polarity(text):
        return TextBlob(str(text)).sentiment.polarity

    with st.spinner("Analysing sentiments..."):
        df_products["Sentiment"] = df_products["Description"].apply(
            get_sentiment
        )
        df_products["Polarity"] = df_products["Description"].apply(
            get_polarity
        )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Positive Products",
        f"{(df_products['Sentiment'] == 'Positive').sum():,}"
    )
    col2.metric(
        "Neutral Products",
        f"{(df_products['Sentiment'] == 'Neutral').sum():,}"
    )
    col3.metric(
        "Negative Products",
        f"{(df_products['Sentiment'] == 'Negative').sum():,}"
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            df_products,
            names="Sentiment",
            title="Sentiment Breakdown",
            color="Sentiment",
            color_discrete_map={
                "Positive": "green",
                "Neutral": "grey",
                "Negative": "red"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top_pos = df_products.sort_values(
            "Polarity", ascending=False
        ).head(10)
        fig2 = px.bar(
            top_pos,
            x="Polarity",
            y="Description",
            orientation="h",
            title="Top 10 Most Positive Products",
            color="Polarity",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Live Review Analyser")
    user_review = st.text_area("Type a customer review to analyse:")
    if user_review:
        polarity = TextBlob(user_review).sentiment.polarity
        sentiment = get_sentiment(user_review)
        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment} (Polarity: {polarity:.2f})")
        elif sentiment == "Negative":
            st.error(f"Sentiment: {sentiment} (Polarity: {polarity:.2f})")
        else:
            st.info(f"Sentiment: {sentiment} (Polarity: {polarity:.2f})")
