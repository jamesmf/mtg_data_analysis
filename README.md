# Modeling 17Lands' MID Premier Draft Game Data


The data that [17 Lands releases](https://www.17lands.com/public_datasets) contains a wealth of information about the format and always generates some interesting conversation. After reading a [few](https://mtgazone.com/17lands-where-i-agree-disagree-and-what-data-you-should-be-taking-from-it/) [posts](https://mtgazone.com/17lands-in-defense-of-the-data/) about the value of the MID Card Rating data, I got interested in modeling the data a bit myself in order to see what nuances might be hard to capture by aggregating a single metric across the dataset. Exploring those subtler dynamics requires capturing the context of each card: the deck it's being played in, the format, the ranking, etc. 

This post is an attempt to use machine learning to understand a bit about what decks are likely to succeed in MID. The goal is not causal inference (I don't aim to say "this deck wins because it plays `Adeline`) but rather to generate some hypotheses and give a framework for thinking about the format. When possible, I'll keep the machine learning (ML) background to a minimum, but if you're interested, the code is on github at the link above.


### A Note on Win Likelihood

The 17Lands platform is great for improving your play, but the claim that the win rate observed in the data was because the average 17Lands user was better than average didn't seem like it was the full story. At the top levels of play this might hold true, but decent matchmaking should then theoretically just pair 17Lands player with higher-quality opponents.

If we instead think about the nature of the draft, the way we observe the data should naturally tilt towards seeing more wins than losses. Higher variance among decks will make this more pronounced. Imagine the most extreme case, with two 17Lands players `A` and `B` playing two non-17Lands opponents `C` and `D`. Player `A` always beats `C` and Player `B` always loses to `D`. After one iteration, in our observed dataset we'll see 7 wins and 3 losses, or a win rate of `0.70`. That is the case with maximum variance - one deck had a win rate of `1.0` and the other `0.0` and we had nothing in between. In reality decks will have many different win rates between `0.0` and `1.0`. For the following analysis, we can make the  assumption that a deck's win likelihood going into a draft is normally distributed around some mean. 

We can then simulate drafts by generating random deck win likelihoods and the resultant number of wins/losses. For each mean and standard deviation of this `deck_win_likelihood` distribution, we can get an observed win rate. We're looking to compare that to the observed win likelihood in our real 17Lands dataset to get an idea of how much above `0.50` the decks a 17Lands user tend to be. Note that the assumption that a deck's win likelihood is constant through a draft is not a strong one, since presumably ranking up along the way changes your opponent pool.

Plotting out various means and standard deviations of this distribution, we get something like this:

![observed wins as a function of win likelihood distribution](/img/win_likelihood_distribution.png)

You can see that if the true distribution has a mean as low as `0.52` and a reasonably high variance of 0.2 (meaning about 67% of decks fall between `win_likelihood (0.32, 0.72)`), we could observe a win likelihood of `0.56` as we do in this dataset. If the variance is much lower, the mean could be closer to `0.545`. Without perfect draft data (every game for every deck), it's hard to know which of these is closer.

While those numbers are lower than the original appearance of a six percent bump to win rate, they're still nothing to be ignored!


### The First Model

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



## Part Two: Decks as Graphs

After trying an attention-based model on the data above, I started wondering if it was more efficient to encode a bit more domain knowledge into the shape of the input to our model. Graph Neural Networks operate over an arbitrary-shape graph, and can do a great job of explicitly capturing relationships between parts of your inputs. For this second half of the post, I've switched to `pytorch` rather than `tensorflow` for ease of using `dgl`, a graph ML package. 


### Modeling the Input

One option to increase the information available to our model would have been to add features to the cards - we could give the model data about each card (its `cmc`, `type`, etc). and expect it to learn reasonable relationships between those inputs. With graph neural networks (GNNs) we can encode those relationships much more explicitly by defining entities/nodes and relationships/edges between them. For those unfamiliar with graphs, some classic datasets you would represent this way are highly relational datasets like business data (`employee_A reports_to employee_B`, `employee_B works_in office_C`) or a social network (`user_A follows user_B`, `user_B <3 content_C`).

By setting up our magic decks this way, we can control the flow of information in our network, we can mix deck-level information with card-level information, and we can have a very flexible data model.

Our decks have the following entities/nodes:

 - deck node - this represents the deck itself. All cards in the deck have an edge to this node
 - sideboard node - this represents the sideboard. All cards in the sideboard have an edge to this node
 - virtual node - this node is connected only to the deck and sideboard nodes. It is meant to collect information from both
 - card nodes - each card is added (one or more times) to the graph, and connected to either the deck or sideboard accordingly
 - core type nodes - each type (creature, planeswalker, sorcery, ...) gets a node, including lands
 - subtype nodes - each word in a type line also gets a node. This lets us model shared types (i.e. a `human wizard` card gets connected to a `human` node and a `wizard` node)
 - cost nodes - each converted mana cost up to 6 gets its own node, allowing us to explicitly learn information about counts of 1-, 2-, ... 6+-drops in the deck

While I don't love python's options for plotting graphs interactively (if you have a package you love for it, post it as an github issue!), and graphs this interconnected are busy to look at, here's an example `MID` deck:

![Example of a deck graph](/img/example_deck_graph.png)

Notice that the degree (the number of edges in/out of a node) provides our model with a way of measuring counts of important features - the more 1-drops the deck has, the more information from the `cmc-1` node will flow to the `deck`/`virtual` node. Similarly, we capture a few other aspects that may be important in some decks: for instance, we have several `human`, `wizard`, and `zombie` cards. These nodes will help cards of similar types share information as well.

We also put some data on the nodes themselves. We create a string that includes `mana_cost`, `type_line`, `card_text`, and `power/toughness` using the `Scryfall/oracle` dataset. This provides a way to model synergies like `Skaab Wrangler` - the text features captures the concept of a `zombie` synergy, and connection to other `zombie` cards through the appropriate type node lets that information flow to those other cards in 2 hops.