FROM python:3.9.6-slim-buster

COPY ./ /credit_card_customer_churn
WORKDIR /credit_card_customer_churn

RUN apt-get update && \
    pip install -r requirements.txt && \
    mkdir -p credit_card_customer_churn/logs &&\
    mkdir -p credit_card_customer_churn/models &&\
    mkdir -p credit_card_customer_churn/images/EDA &&\
    mkdir -p credit_card_customer_churn/images/results

ENTRYPOINT ["python"]
CMD ["credit_card_customer_churn"]
