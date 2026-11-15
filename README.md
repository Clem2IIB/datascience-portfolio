# Data Science Portfolio

This repository gathers the data science projects I worked on while learning Python, pandas, scikit-learn and data visualization. Most of them started as DataCamp projects. I then extended each one with my own exploration, models and checks, and wrote it up as a full notebook: context, goal, exploration, modelling, results and limits.

Each project lives in its own folder, with its notebook and a `data/` folder. The notebooks are saved with their outputs, so they can be read directly on GitHub.

## Projects

### Machine Learning

- [Will a Netflix Title Stay in the Top 10?](netflix-top10-ranking-prediction/netflix_top10_ranking_prediction.ipynb): Predicting from its first week whether a title stays in the global top 10 a month later, and grouping 94 countries by their viewing tastes.
- [Which Recipes Bring Traffic to the Homepage?](recipe-traffic-classification/recipe_traffic.ipynb): Choosing the homepage recipes that bring high traffic to a recipe website, for the DataCamp Data Scientist Professional certification.
- [Predicting How Long Customers Rent DVDs](movie-rental-duration-regression/movie_rental_duration.ipynb): Predicting how many days a customer keeps a rented DVD, comparing four regression models.
- [Choosing the Best Crop from Soil Measures](crop-selection-classification/crop_selection.ipynb): Recommending a crop from four soil measures, and finding the single measure that predicts it best.
- [Clustering Penguins in Antarctica](penguins-clustering/penguins_clustering.ipynb): Grouping penguins by their physical measurements with K-means, without knowing their species.

_Tools: pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn, plotly_

### RAG

- [Shakespeare Chatbot](rag-shakespeare-chatbot/): A chatbot that answers questions about Shakespeare's complete works using only passages it retrieves from the text, with an [evaluation notebook](rag-shakespeare-chatbot/rag_evaluation.ipynb) that measures how often it finds the right passage.

_Tools: ChromaDB, sentence-transformers, BM25, LM Studio, Claude API, Streamlit_

The chatbot is an app, not a notebook, and it has its own installation. See its [README](rag-shakespeare-chatbot/README.md) to install and use it.

### Data Analysis and Visualisation

- [Visualizing the History of Nobel Prize Winners](nobel-winners-eda/nobel_winners_eda.ipynb): How the profile of Nobel Prize winners changed from 1901 to 2023, by gender, birth country, age and shared prizes.

_Tools: pandas, matplotlib, seaborn_

## Running the notebooks

The library versions are the ones I used, so the results should match the saved outputs. Each notebook reads its data with a relative path (`data/...`), so it should be run from its own folder. These steps do not install the chatbot (see its README).

**With pip and a virtual environment** (needs Python 3.13 or later):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

**With [uv](https://docs.astral.sh/uv/)**, which installs the exact versions saved in `uv.lock`:

```bash
uv sync
uv run jupyter notebook
```

## A note on AI use

Each notebook starts with a short note on how I used AI (Claude) for that project.

The idea and structure of this portfolio is inspired by [Sajal Sharma's data science portfolio](https://github.com/sajal2692/data-science-portfolio).
