package org.grouplens.mooc.cbf;

import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.pref.Preference;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.List;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class TFIDFItemScorer extends AbstractItemScorer {
    private final UserEventDAO dao;
    private final TFIDFModel model;
    private final CosineVectorSimilarity cosineVectorSimilarity;

    /**
     * Construct a new item scorer.  LensKit's dependency injector will call this constructor and
     * provide the appropriate parameters.
     *
     * @param dao The user-event DAO, so we can fetch a user's ratings when scoring items for them.
     * @param m   The precomputed model containing the item tag vectors.
     */
    @Inject
    public TFIDFItemScorer(UserEventDAO dao, TFIDFModel m) {
        this.dao = dao;
        model = m;
        cosineVectorSimilarity = new CosineVectorSimilarity();
    }

    /**
     * Generate item scores personalized for a particular user.  For the TFIDF scorer, this will
     * prepare a user profile and compare it to item tag vectors to produce the score.
     *
     * @param user   The user to score for.
     * @param output The output vector.  The contract of this method is that the caller creates a
     *               vector whose possible keys are all items that should be scored; this method
     *               fills in the scores.
     */
    @Override
    public void score(long user, @Nonnull MutableSparseVector output) {
        // Get the user's profile, which is a vector with their 'like' for each tag
        // SparseVector userVector = makeUserVector(user);
        SparseVector userVector = makeWeightedUserVector(user);

        // Loop over each item requested and score it.
        // The *domain* of the output vector is the items that we are to score.
        for (VectorEntry e: output.fast(VectorEntry.State.EITHER)) {
            // Score the item represented by 'e'.
            // Get the item vector for this item
            SparseVector iv = model.getItemVector(e.getKey());
            // TODO Compute the cosine of this item and the user's profile, store it in the output vector
            double cosine = cosineVectorSimilarity.similarity(userVector, iv);
            // double cosine = userVector.dot(iv) / (userVector.norm() * iv.norm());
            output.set(e.getKey(), cosine);

            // TODO And remove this exception to say you've implemented it
            // throw new UnsupportedOperationException("stub implementation");
        }


  }


    /**
     * Using formula
     *     u⃗ =∑i∈(u)(rui−μu)i⃗
     *
     * u⃗  ->  User Vector
     * i⃗  ->  Item Vector
     * I(u) -> Set of items rated by user
     * ut, It -> User u's or item i's score for tag
     * rui -> Ratings of user U for Item i
     * μu  -> Average of user's ratings
     *
     * @param user
     * @return
     */
    private SparseVector makeWeightedUserVector(long user) {
        List<Rating> userRatings = dao.getEventsForUser(user, Rating.class);

        MutableSparseVector profile = model.newTagVector();
        profile.fill(0);

        double averageRating = 0;
        int numRatings = 0;
        //calcuate average of user ratings
        for (Rating rating : userRatings) {
            averageRating += rating.getPreference().getValue();
            numRatings++;
        }
        averageRating = averageRating/numRatings;

        //for each Item User has rated
        for (Rating r: userRatings) {
            if (r.getPreference() != null) {
                double rating = r.getPreference().getValue();
                if (rating > 0) {
                    double netWeight = (rating - averageRating);
                    SparseVector itemVector =  model.getItemVector(r.getPreference().getItemId());
                    MutableSparseVector mutableItemSparseVector = itemVector.mutableCopy();
                    mutableItemSparseVector.multiply(netWeight);
                    profile.add(mutableItemSparseVector);
                }
            }
        }
        return profile.freeze();

    }

    private SparseVector makeUserVector(long user) {
        // Get the user's ratings
        List<Rating> userRatings = dao.getEventsForUser(user, Rating.class);
        if (userRatings == null) {
            // the user doesn't exist
            return SparseVector.empty();
        }

        // Create a new vector over tags to accumulate the user profile
        MutableSparseVector profile = model.newTagVector();
        // Fill it with 0's initially - they don't like anything
        profile.fill(0);

        // Iterate over the user's ratings to build their profile
        for (Rating r: userRatings) {
            // In LensKit, ratings are expressions of preference
            Preference p = r.getPreference();
            // We'll never have a null preference. But in LensKit, ratings can have null
            // preferences to express the user unrating an item
            if (p != null && p.getValue() >= 3.5) {
                // The user likes this item!
                // TODO Get the item's vector and add it to the user's profile
                //get vector for item which user has liked
                profile.add(model.getItemVector(p.getItemId()));
            }
        }

        // The profile is accumulated, return it.
        // It is good practice to return a frozen vector.
        return profile.freeze();
    }
}












































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































