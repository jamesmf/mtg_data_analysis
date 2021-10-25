# Modeling 17Lands' MID Premier Draft Game Data


The data that [17 Lands releases](https://www.17lands.com/public_datasets) contains a wealth of information about the format and always generates some interesting conversation. After reading a [few](https://mtgazone.com/17lands-where-i-agree-disagree-and-what-data-you-should-be-taking-from-it/) [posts](https://mtgazone.com/17lands-in-defense-of-the-data/) about the value of the MID Card Rating data, I got interested in modeling the data a bit myself in order to see what nuances might be hard to capture by aggregating a single metric across the dataset. Exploring those subtler dynamics requires capturing the context of each card: the deck it's being played in, the format, the ranking, etc. 

This post is an attempt to use machine learning to understand a bit about what decks are likely to succeed in MID. The goal is not causal inference (I don't aim to say "this deck wins because it plays `Adeline`) but rather to generate some hypotheses and give a framework for thinking about the format. When possible, I'll keep the machine learning (ML) background to a minimum, but if you're interested, the code is on github at the link above.


### The Model

When analyzing data or training models, one of the most important considerations is the "shape" of your data. What is a single unit or example? In this case, the `17 Lands` dataset is naturally at the game level - we get information about what is in the deck/sideboard, whether the user won or lost, a bit of user information, and a bit of draft information. One appealing option to proceed would be to roll this information up to the user-draft level and model the distribution of how many wins a user will get in a draft. However, that approach requires you to be sure you aren't missing any games for a user within a draft, and you would likely need to make the input to your model simply all the cards they drafted, rather than the specific ones they played per game. If we consider a specific user's deck to be a combination of `(draft_id, main_colors, user_win_rate_bucket)`, then about 11% of combinations don't meet the criterion of having either 7 wins or 3 losses. With that in mind, the model is set up to accept the deck and sideboard for a specific game. 

One comment about the utility of this data was that cards may play very differently in different tiers of play. To address this, the model also accepts the user's `rank` at the time of the game. 

The output of the model is simply a prediction of whether the deck will win the game in question. The result is a model whose signature is `win_likelihood = model(deck, sideboard, rank)`. This signal will be very noisy, as even a great deck will lose plenty, but the dataset is reasonably large, and hopefully this comes out in the wash.


### The Training

The model is trained on 90% of drafts, holding 10% of drafts out for us to analyze. This helps us avoid [overfitting](https://en.wikipedia.org/wiki/Overfitting), though it can't help us with the fact that our data is biased by being collected only by users of `17 Lands`. The mean win rate is around `0.56`, so we are hoping the accuracy of the model on this holdout set is greater than that. Theoretically players are being matched up against fairly similar opponents, and we know nothing about the opponent's deck, so it would be surprising if our deck alone let us predict win likelihood with extremely high accuracy.

In practice, the accuracy on the holdout set tends to hover around `0.62` for our simple model. To validate it, we can plot the calibration curve. This confirms that when the model predicts a deck has about a 50% chance of winning, we see about 50% wins; when we predict 80%, we observe about 80%, etc.

![calibration curve](/img/calibration_curve.png)

We can also inspect the prediction distribution, which is also encouraging. Given the limited information we're feeding the model, the majority of predictions are that a deck will win about 35-75% of the time.

![plot of distribution of predictions](/img/distribution_of_predictions.png)

With those reassurances, let's start exploring what the model can tell us!


### Decks 

First, we can look at real decks from the holdout set that the model thinks will have very high win likelihoods. Here are a few examples:

![Example of a strong Dimir deck](/img/example_deck_good_dimir.png)

![Example of a strong Simic deck](/img/example_deck_good_simic.png)

We can also check out an example of a deck the model believes to be more or less doomed:

![Example of a weak Dimir deck](/img/example_deck_poor_dimir.png)


### Variance

Our model lets us explore not just the raw win rates for decks, but the distributions of predicted win probabilities, which contain more information than aggregate values. For instance, the blog linked at the beginning called out (Azorius, Dimir, Simic) as the top three two-color combinations, with very similar win rates (59.3, 58.4, 58.3). If we plot the predicted win distributions of those, we see some very meaningful differences.

![Plot of predicted win rates for Azorius, Dimir, Simic](/img/win_rates.png)

While the mean of the distribution for Azorius is the highest, we can see that Dimir gets far more play. So by raw number of times that players have assembled a deck our model believes can win 70% of games, Dimir has a huge advantage.

Simic shows a tighter distribution - very rarely do we see a Simic deck our model thinks will win 70% or more of its games. But not only is the peak of the distribution lower, it is yet still less drafted than Azorius, suggesting it's even less likely you'll come out of a draft with a killer Simic deck.