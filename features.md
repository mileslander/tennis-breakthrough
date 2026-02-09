# Tennis Breakthrough Project Predictive Variables

An inherently difficult prediction requires strong predictive variables to measure several independent signals to find meaning in player performance and distinguish noise from potential. Several rounds of feature engineering were carried out to achieve high correlation with the breakthrough tiers. The rounds of features engineering are explained below

## Initial Features

The initial predictive variables:

- 4 month win rate
- win count against players ranked 25 spots higher
- win rate against players ranked 25 spots higher
- win count against players ranked 50 spots higher
- win rate against players ranked 50 spots higher
- win count against players ranked 100 spots higher
- win rate against players ranked 100 spots higher

These variables represent ceiling and floor skill levels. An unrestricted win rate represents a player's floor. Constrained win rates and counts represent a player's ceiling, for example in a match against a player ranked 25 spots higher.

- An unrestricted rate under 0.5 suggests an uncompetitive player, very unlikely to breakthrough.  
- An unrestricted rate above 0.7 suggest a highly competitive player, a breakthrough candidate.
- Constrained rates above 0.5 suggest an ability to compete at a higher level, a breakthrough candidate.
- A win count against players ranked 25 spots higher greater than 7 suggest an ability to compete at a higher level, a breakthrough candidate.

Analysis shows that the best feature of this group, _4 month win rate_, has a 0.1736 correlation with breakthrough and almost all of the ceiling features have greater than 0.85 multicolinearity. The high multicolinearity and low correlation with breakthrough of these predictive variables clearly suggest that improved predictive variables are needed

## General Feature Engineering

This round prioritized independent categories of predictive variables in reaction to the initial set all being of the same category. Variables of the same category are likely to have high levels of multicolinearity which makes them redundant for a Logistic Regression model. LR models struggle to determine the effect of individual independent variable when several feature communicate the same signal.

The first round of feature engineering added the following predictive variables.

- Demographic Features
  - Age
  - Nationality
  - Dominant Hand
  - Height
- Match Psychology Features
  - Average Win Duration
  - Average Loss Duration
  - Fight Score (Average Loss Duration / Number of Losses)
  - Dominance Score (Average Win Duration / Number of Wins)
- Performance Interaction Features
  - Consistency Volume (win count against players ranked 25 spots higher * 4 month win rate)
  - Quality Consistency (win rate against players ranked 25 spots higher * 4 month win rate)
  - Win Concentration (win count against players ranked 25 spots higher / win count against players ranked 100 spots higher + 1)
  - Rate Progression (win rate against players ranked 25 spots higher / win rate  against players ranked 100 spots higher)
  - Elite Indicator (win count against players ranked 25 spots higher >= 7 and 4 month win rate >= 0.6)
  - Quality Wins High (win count against players ranked 25 spots higher >= 8)
  - Rate Stability (win rate against players ranked 25 spots higher - win rate against players ranked 100 spots higher)
  - Volume Dominance (win count against players ranked 25 spots higher - win count against players ranked 100 spots higher)

 Match psychology features test a player's determination to win, players who give up easily won't win as many matches, hence a worse breakthrough candidate. _Fight score_ has over a 0.17 correlation with breakthrough.

 Demographic features were tested although had very low correlation levels with breakthrough. _Height_ has under 0.05 correlation with breakthrough.

 Composite features tested transformations of the initial features, these worked very well. _Volume dominance_ has a 0.40 correlation with breakthrough.

 This is a major improvement but just two or three strong features is not good enough for an accurate model.

## Feature Engineering Round Two

Using the new features another batch of composite predictive variables was engineered:

- Skill + Mental Toughness Features
  - Champion Score (win count against players ranked 25 spots higher * (fight score / 100))
  - Elite Fighter (win count against players ranked 25 spots higher >= 7 and fight score >= median fight score)
  - Tough Consistency ((fight score / 100) * 4 month win rate)
- Sustained Mental Toughness
  - Total Matches Estimate (win count against players ranked 25 spots higher / win rate against players ranked 25 spots higher)
  - Sustained Toughness (Total Matches Estimate * (fight score / 100))
  - Complete Warrior (Total Matches Estimate _4 month win rate_ (fight score / 100))
  - Volume at Right Level (Total Matches Estimate * (Volume Dominance / 10))
- Tournament Strategy Features
  - Strategic Dominance (volume dominance * win concentration)
  - Plays Right Level (volume dominance >= 5 and win concentration >= 2.0)
  - Tournament Efficiency (win count against players ranked 25 spots higher / (win count against players ranked 25 spots higher + win count against players ranked 100 spots higher + 1))
- Multi-Signal Combination Features
  - Breakthrough Composite (((volume dominance / 10) + (win concentration / 3) + 4 month win rate) / 3)
  - Elite Multi Gate (volume dominance >= 5 and win count against players ranked 25 spots higher >= 7 and 4 month win rate >= 0.55)
  - Peak Performer (Consistency Volume * Volume Dominance)
- Efficiency and Rate Based Features
  - Level Efficiency (win rate against players ranked 25 spots higher * win concentration)
  - Quality Density (win count against players ranked 25 spots higher / (win count against players ranked 25 spots higher + win count against players ranked 50 spots higher + win count against players ranked 100 spots higher + 1))
  - Dominant Consistency (volume dominance * 4 month win rate)

Skill + Mental Toughness features represent a player's determination to win and their results. A player who competes hard at all times and is a winning player is a breakthrough candidate. _Tough consistency_ has over a 0.45 correlation with breakthrough.

Sustained mental toughness features represent a player's determination to win combined with their match volume. A player competing hard in a few tournaments may be promising, while a player competing hard with a full schedule is a strong breakthrough candidate. _Complete warrior_ has a 0.31 correlation with breakthrough.

Tournament Strategy Features represent if a player is playing at the right level. A player who is dominating their level is a strong breakthrough candidate. _Strategic Dominance_  has a 0.38 correlation with breakthrough.

Multi-signal combination and efficiency and rate based features combine previously engineered features and the initial set of feature. _Breakthrough composite_ has a 0.33 correlations with breakthrough.

These features were a major improvement but not yet correlated enough with breakthrough to model with. Player's initial ranking was not included in these predictive variables which was a major oversight.

## Final Feature Engineering

A final set of features was added using ranking data:

- Final Window Rank (Rank in last month of data collection window)
- Initial Rank Inverted (1000 / initial rank + 1)
- Ranking Velocity ((initial rank - final rank) / 4)
- Ranking Percent Change (((initial rank - final rank) / initial rank) * 100)
- Performance vs Baseline (win count against players ranked 25 spots higher / ((initial rank + 1) / 100))
- Velocity vs Baseline (ranking velocity / ((initial rank + 1) / 100))
- Warrior Adjusted (Complete Warrior *  (1000 / (initial rank + 1)))
- Gap to Elite (Initial Rank - 100)
- Already Competitive (Baseline Rank < 500x)
- Breakthrough Feasibility ((1000 / (initial rank + 1)) _complete warrior_ (ranking velocity / 51))

A players chance at breaking through is in large part decided by their initial ranking. With just a year to breakthrough a player ranked 250 has a much better chance than a player ranked 1500. This feature set is much stronger than the rest, 4 of the top 5 feature are from this round. _Initial Rank Inverted_ has the highest correlation with breakthrough at 0.734!
