/**
 * AllValuesIterator.java
 * Written by Gil Tene of Azul Systems, and released to the public domain,
 * as explained at http://creativecommons.org/publicdomain/zero/1.0/
 *
 * @author Gil Tene
 */

package org.HdrHistogram_voltpatches;

import java.util.Iterator;

/**
 * Used for iterating through histogram values using the finest granularity steps supported by the underlying
 * representation. The iteration steps through all possible unit value levels, regardless of whether or not
 * there were recorded values for that value level, and terminates when all recorded histogram values are exhausted.
 */

public class AllValuesIterator extends AbstractHistogramIterator implements Iterator<HistogramIterationValue> {
    int visitedSubBucketIndex;
    int visitedBucketIndex;

    /**
     * Reset iterator for re-use in a fresh iteration over the same histogram data set.
     */
    public void reset() {
        reset(histogram);
    }

    private void reset(final AbstractHistogram histogram) {
        super.resetIterator(histogram);
        visitedSubBucketIndex = -1;
        visitedBucketIndex = -1;
    }

    /**
     * @param histogram The histogram this iterator will operate on
     */
    public AllValuesIterator(final AbstractHistogram histogram) {
        reset(histogram);
    }

    @Override
    void incrementIterationLevel() {
        visitedSubBucketIndex = currentSubBucketIndex;
        visitedBucketIndex = currentBucketIndex;
    }

    @Override
    boolean reachedIterationLevel() {
        return (visitedSubBucketIndex != currentSubBucketIndex) || (visitedBucketIndex != currentBucketIndex);
    }
}
