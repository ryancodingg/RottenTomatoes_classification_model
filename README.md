

# Motivation

The motivation behind this project is to build a machine learning model that can predict the Rotten Tomatoes rating of movies based on critic reviews. By utilizing two large datasets, namely rotten_tomatoes_critic_reviews.csv and rotten_tomatoes_movies.csv obtained from Kaggle, I aim to develop a classification algorithm that can effectively predict the rating of movies.

Rotten Tomatoes is a popular online platform that provides movie reviews and ratings. The ratings provided on Rotten Tomatoes are determined by a combination of critics' reviews and audience ratings. Critics' reviews play a significant role in influencing the overall rating assigned to a movie. Therefore, by analyzing critic reviews, we can gain insights into the sentiment and opinions expressed by critics, which can subsequently be used to predict the movie's Rotten Tomatoes rating.

# Dataset Description

I use 2 dataset: 
*  `rotten_tomatoes_critic_reviews.csv`
    - `rotten_tomatoes_link`: each link will associate with an unique ID of a movie
    - `critic_name`: name of critic
    - `top_critic`: tomatometer-approved critic: True or False
    - `publisher_name`: name of Publisher
    - `review_type`: Fresh, Rotten or Certified Fresh 
    - `review_score`: score for the movie
    - `review_date`: date of review
    - `review_content`: content of the review
* `rotten_tomatoes_movies.csv`
    - `rotten_tomatoes_link`: each link will associate with an unique ID of a movie
    - `movie_title`: name of the movie
    - `movie_info`: brief introducrion of the movie
    - `critic_consensus`: rotten Tomatoes's comment
    - `content_rating`: rating of content - PG, R, NR, PG-13,G 
    - `genres`: type of movie
    - `directors`: name of directors
    - `authors`: name of authors
    - `actors`: list of actors
    - `original_release_date`: date which first made available to public
    - `streaming_release_date`: for only streaming providers
    - `runtime`: length of movie
    - `production_company`: name of production house
    - `tomatometer_status`: Fresh, Rotten or Certified Fresh
    - `tomatometer_rating`: percentage of positive reviews
    - `tomatometer_count`: critic ratings counted for the calculation of the tomatomer status
    - `audience_status`: label assigned Spill or Upright
    - `audience_rating`: percentage of positive rating users
    - `audience_count`: total rating audience
    - `tomatometer_top_critics_count`: number of rating by top critics
    - `tomatometer_fresh_critics_count`:  number of critic ratings labeled "Fresh"
    - `tomatometer_rotten_critics_count`: number of critic ratings labeled "Rotten" 
