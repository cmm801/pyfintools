""" Data structures for working with OHLC data.

"""
import numpy as np
import pandas as pd
import heapq

import matplotlib.pyplot as plt


DEFAULT_ABS_TOL = 1e-6
DEFAULT_REL_TOL = 1e-6

# Default minimum distance between what we count as 'distinct' lines
DEFAULT_LINE_SEPARATION = 10

class Line:
    def __init__(self, point, timedelta=1, slope=None, point2=None):
        x, y = point
        self.timedelta = timedelta

        if (slope is not None) == (point2 is not None):
            raise ValueError('Exactly one of "slope" or point2 must be specified.')
        elif slope is not None:
            self.slope = slope
        elif point2 is not None:
            x2, y2 = point2
            dy = y2 - y
            dx = (x2 - x) / self.timedelta
            self.slope = dy / y
        else:
            assert False, 'Should not ever reach this block'

        self.x = x
        self.y = y
        
    def get_x_vals(self, y_vals):
        dy = y_vals - self.y

        x_vals = self.x + (dy/self.slope) * self.timedelta
        
        return x_vals

    def get_y_vals(self, x_vals):
        dx = (x_vals - self.x) / self.timedelta
        y_vals = self.y + self.slope * dx
        return y_vals
    
    def plot(self, x_vals=None, y_vals=None, ax=None, **kwargs):
        if x_vals is not None and y_vals is not None:
            raise ValueError('Only one of x_vals or y_vals can be provided.')
        elif x_vals is not None:
            y_vals = self.get_y_vals(x_vals)
        elif y_vals is not None:
            x_vals = self.get_x_vals(y_vals)

        if ax is not None:
            ax.plot(x_vals, y_vals, **kwargs)
        else:
            plt.plot(x_vals, y_vals, **kwargs)

    def calc_distance(self, x0, y0):
        y_line = self.y + self.slope * (x0 - self.x) / self.timedelta
        dist = np.abs(y_line - y0)
        return dist

    def is_crossing_segment(self, point1, point2, rtol=None):
        if rtol is None:
            rtol = DEFAULT_REL_TOL

        x1, y1 = point1
        x2, y2 = point2
        sign1 = np.sign(self.get_y_vals(x1) - y1)
        sign2 = np.sign(self.get_y_vals(x2) - y2)
        return sign1 != sign2

    def is_point(self, x, y, rtol=None):
        if rtol is None:
            rtol = DEFAULT_REL_TOL

        y_line = self.get_y_vals(x)
        return np.isclose(y_line, y, rtol=rtol)


class TrendlineHelper:
    def __init__(self, ohlc, sep=None, atol=None, rtol=None, min_len=1):        
        self.ohlc = ohlc
        self.sep = sep
        self.atol = atol
        self.rtol = rtol
        self.min_len = min_len

    @property
    def ohlc(self):
        return self._ohlc
    
    @ohlc.setter
    def ohlc(self, ts):        
        self._ohlc = ts

        # Reset some calculated variables
        self._expanding_min = None
        self._expanding_max = None        

    @property
    def sep(self):
        """ Getter for the minimum separation between lines (in seconds)"""
        return self._sep

    @sep.setter
    def sep(self, s):
        """ Setter for the minimum separation between lines (in seconds)"""
        if s is None:
            self._sep = pd.Timedelta(DEFAULT_LINE_SEPARATION, self.ohlc.frequency).total_seconds()
        elif isinstance(s, (int, np.float32, float)):
            self._sep = s
        else:
            raise ValueError('Expected numeric input.')

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, a):
        if a is None:
            self._atol = DEFAULT_ABS_TOL
        else:
            self._atol = a

    @property
    def rtol(self):
        return self._rtol

    @rtol.setter
    def rtol(self, r):
        if r is None:
            self._rtol = DEFAULT_REL_TOL
        else:
            self._rtol = r

    @property
    def expanding_min(self):
        if self._expanding_min is None:
            self._expanding_min = self.ohlc.low.expanding().min()
        return self._expanding_min

    @property
    def expanding_max(self):
        if self._expanding_max is None:
            self._expanding_max = self.ohlc.high.expanding().max()
        return self._expanding_max

    def get_all_trendlines(self, step_size=1):
        """ Get the upper/lower lines (resistance/support) for a all observation points.

            Arguments:
                ohlc: an OHLC time series object.
                step_size: how many obsevations for which we should calculate the
                    trendlines. By default, step_size is 1 which means we find
                    trendlines at every x-value. If step_size were set to 10, then 
                    we would calculate trendlines instead at every 10th observation.

            Returns:
                a DataFrame containing the upper/lower trendlines info.
        """
        line_list = []
        for idx in range(1, self.ohlc.index.size, step_size):
            x_R = self.ohlc.index[idx]
            lines = self.get_trendlines_for_single_point(x_R)
            line_list.append(lines)

        all_lines = pd.concat(line_list, axis=0).reset_index(drop=True)
        return all_lines

    def get_trendlines_for_single_point(self, x_R):
        """ Get the upper/lower lines (resistance/support) for a single input point.

            Arguments:
                ohlc: an OHLC time series object.
                x_R: (Timestamp) the last time for which all lines are calculated.

            Returns:
                a DataFrame containing the upper/lower trendlines info.
        """
        lower_lines = self._find_all_tangent_lines_for_single_point(-1, x_R)
        upper_lines = self._find_all_tangent_lines_for_single_point(+1, x_R)

        upper_lines['direction'] = +1
        lower_lines['direction'] = -1

        return pd.concat([lower_lines, upper_lines], axis=0)

    def _find_all_tangent_lines_for_single_point(self, direction, x_R):
        """ Find all of the upper or lower tangent lines from a given point.

            The algorithm starts from the point x_R and moves backward through historical 
            points (x_L). It finds the upper/lower lines that represent the convex hull
            of the time series ts in the interval [x_L, x_R].

            Arguments:
                direction: (int) specifies whether we find upper (+1) or lower (-1) 
                    tangent lines.
                x_R: (Timestamp) the last time for which all lines are calculated.
        """
        if x_R not in self.ohlc.index:
            raise ValueError('The target point cannot be found in the time series index.')

        if direction == 1:
            ts = self.ohlc.high[:x_R]
            expanding_vals = self.expanding_max.values
        elif direction == -1:
            ts = self.ohlc.low[:x_R]
            expanding_vals = self.expanding_min.values
        else:
            raise ValusError('Argument "direction" must be either +1 or -1.')

        # Initialise some containers to store results
        index_vals = []
        slope_vals = []
        intercept_vals = []
        x_L_vals = []
        x_R_vals = []    
        y_L_vals = []
        y_R_vals = []
        tstmps = []
        
        # Precalculate some helpful values
        y_R = ts.values[-1]
        idx_R = np.where(x_R == ts.index)[0][0]
        dx_vals = self.ohlc.timestamps[idx_R] - self.ohlc.timestamps[:idx_R+1]
        dy_vals = ts.values[idx_R] - ts.values

        # Initialize variable for the slope of the bounding line
        slope_bound = direction * np.inf        
        
        # Loop in reverse starting from the last to 2nd observation
        for idx_L in reversed(range(1, idx_R)):
            x_L = ts.index[idx_L]
            y_L = ts.values[idx_L]
            dx = dx_vals[idx_L]

            # Determine whether we have already found all lines
            y_L_line = y_R - slope_bound * dx

            # Break from loop if no more tangent lines can exist without
            #  intersecting some part of the price curve
            if idx_L < idx_R - 1:
                if direction == 1:
                    if slope_bound < 0 and expanding_vals[idx_L] <= y_L_line:
                        break
                elif direction == -1:
                    if slope_bound > 0 and expanding_vals[idx_L] >= y_L_line:
                        break                        

            # Determine whether the new point tightens the bound
            dy = dy_vals[idx_L]
            slope = dy / dx        
            if direction == 1:
                is_tighter_bound = slope < slope_bound - self.atol
            else:
                is_tighter_bound = slope > slope_bound + self.atol

            if is_tighter_bound:
                if dx >= self.min_len - self.atol:
                    index_vals.append(x_L)
                    slope_vals.append(slope)
                    x_R_vals.append(x_R)
                    x_L_vals.append(x_L)
                    y_R_vals.append(y_R)            
                    y_L_vals.append(y_L)
                    tstmps.append(self.ohlc.timestamps[idx_L])
                    
                    intercept = y_R - slope * dx_vals[0]
                    intercept_vals.append(intercept)

                # Update the slope of the tightest bound
                slope_bound = slope

        output_vals = np.vstack([slope_vals, 
                                 [ts.index[0]] * len(intercept_vals), intercept_vals, 
                                 x_L_vals, y_L_vals,
                                 x_R_vals, y_R_vals, tstmps]).T
        columns = ['slope', 'x_0', 'y_0', 'x_L', 'y_L', 'x_R', 'y_R', 'timestamp']
        df = pd.DataFrame(output_vals, columns=columns)
                          
        # Sort the values
        df.sort_values('x_L', inplace=True)

        # If 'sep' is specified, make sure peaks/troughs that determine lines are
        #    sufficiently separated from one another
        if df.shape[0]:
            dist = np.hstack([self.sep + 1, 
                              df.timestamp[1:].values - df.timestamp[:-1].values])
            df = df.loc[dist >= self.sep]

        # Discard the 'timestamp' column as it is no longer needed
        df.drop('timestamp', axis=1, inplace=True)
        return df
    
    def plot_lines(self, lines, x_obs, tightness=None, ax=None, xlim=None, ylim=None):
        """ Plot the trendlines for a single point.
        
            Arguments:
                lines: (DataFrame) the information about trendlines for a single point.
                x_obs: (Timestamp) the point for which the trendlines are drawn
                tightness: (float) how wide the x-axis plot range should be (in seconds)
        """
        y_obs = self.ohlc.high[x_obs]/2 + self.ohlc.low[x_obs]/2

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        self.ohlc.high.plot(color='blue', ax=ax)
        self.ohlc.low.plot(color='orange', ax=ax)

        for _, s in lines.iterrows():
            if isinstance(s.x_L, pd.Timestamp):
                _timedelta = pd.Timedelta(seconds=1)
            else:
                _timedelta = 1
                
            L = Line(slope=s.slope, point=(s.x_L, s.y_L), timedelta=_timedelta)

            if s.direction == 1:
                L.plot(x_vals=self.ohlc.index, ax=ax, color='green')
                ax.axvline(L.x, color='green', linestyle='-.')
            else:
                L.plot(x_vals=self.ohlc.index, ax=ax, color='red')
                ax.axvline(L.x, color='red', linestyle='-.')

        # Draw horizontal/vertical lines centered on the target point
        ax.axvline(x_obs, color='black', linestyle='-.')
        ax.axhline(y_obs, color='black', linestyle='-.')

        # Set the plot range for the x-axis
        if tightness is not None:
            x_plot_L = x_obs - pd.Timedelta(seconds=tightness)
            x_plot_R = x_obs + pd.Timedelta(seconds=tightness)
            ax.set_xlim(x_plot_L, x_plot_R)
        else:
            x_plot_L = self.ohlc.index[0]
            x_plot_R = x_obs
        
        # Choose a y-range that adds a bit of padding on the top/bottom
        y_plot_low = self.ohlc.low[x_plot_L:x_plot_R].min()
        y_plot_high = self.ohlc.high[x_plot_L:x_plot_R].max()
        dy = (y_plot_high - y_plot_low)
        y_plot_low -= 0.1 * dy
        y_plot_high += 0.1 * dy
        
        if ylim is None:
            ax.set_ylim(y_plot_low, y_plot_high)
        else:
            ax.set_ylim(ylim)
            
        if xlim is not None:
            ax.set_xlim(xlim)


class FeatureHelper:
    def __init__(self, ohlc_obj):
        self.ohlc_obj = ohlc_obj

    def calculate_features(self, lines, x_obs, width_nbrhood, width_tangent, max_intersections=1):
        """ Compute line features for trend lines at a single point.
        """        
        df_list = []
        for d in [+1, -1]:
            sub_lines = lines.query(f'direction == {d}')
            df_d = self._calculate_features_one_direction(d, sub_lines, x_obs,
                                                         width_nbrhood, width_tangent,
                                                         max_intersections=max_intersections)            
            df_list.append(df_d)

        df = pd.concat(df_list, axis=0)
        return df

    def _calculate_features_one_direction(self, direction, lines, x_obs, 
                                          width_nbrhood, width_tangent, max_intersections=1):
        """ Calculate the features for trend lines at a single point in a single direction. """
        # Only keep lines that were obtained before the current observation
        lines = lines.loc[lines.x_R <= x_obs]

        if direction == 1:
            y_obs = self.ohlc_obj.high[x_obs]
            sub_ts = self.ohlc_obj.high[:x_obs]
        elif direction == -1:
            y_obs = self.ohlc_obj.low[x_obs]
            sub_ts = self.ohlc_obj.low[:x_obs]
        else:
            raise ValueError('Direction must be either +1 or -1.')

        idx_obs = np.where(x_obs == sub_ts.index)[0][0]
        timestamps = self.ohlc_obj.timestamps[:idx_obs+1]
        t_obs = timestamps[idx_obs]

        T = lines.shape[0]
        start_dates = np.array([pd.NaT] * T)
        tangent_rightmost = np.array([pd.NaT] * T)
        tangent_leftmost = np.array([pd.NaT] * T)
        flipped = np.array([False] * T, dtype=bool)
        tangents = np.zeros((T,), dtype=int)
        neighborhoods = np.zeros((T,), dtype=int)
        intersections = np.zeros((T,), dtype=int)

        for idx_line in range(lines.shape[0]):
            s = lines.iloc[idx_line]
            L = Line(slope=s.slope, point=(timestamps[0], s.y_0), timedelta=1)
            if (direction == 1 and L.get_y_vals(t_obs) < y_obs - self.ohlc_obj.atol) \
                    or (direction == -1 and L.get_y_vals(t_obs) > y_obs + self.ohlc_obj.atol):
                # This line has been exceeded at the observation point
                start_dates[idx_line] = x_obs
                flipped[idx_line] = True

            # Pre-compute the distances of all prices to the candidate line
            y_vals_line = L.get_y_vals(timestamps)
            distances = np.abs(y_vals_line - sub_ts.values)
            is_in_line = np.isclose(sub_ts.values, y_vals_line)
            
            idx2 = idx_obs
            idx1 = idx2 - 1
            while idx1 > 0:
                if intersections[idx_line] >= max_intersections:
                    break

                x1 = sub_ts.index.values[idx1]
                x2 = sub_ts.index.values[idx2]
                y1 = sub_ts.values[idx1]
                y2 = sub_ts.values[idx2]
                t1 = timestamps[idx1]
                t2 = timestamps[idx2]                
                d1 = distances[idx1]
                
                if d1 < width_nbrhood:
                    neighborhoods[idx_line] += 1

                if d1 < width_tangent:
                    tangents[idx_line] += 1
                    tangent_leftmost[idx_line] = x1
                    if tangent_rightmost[idx_line] is pd.NaT:
                        tangent_rightmost[idx_line] = x1

                if is_in_line[idx1]:
                    idx1 -= 1
                elif L.is_crossing_segment((t1, y1), (t2, y2)):
                    if start_dates[idx_line] is pd.NaT:
                        start_dates[idx_line] = x2
                    intersections[idx_line] += 1
                else:
                    idx2 = idx1
                    idx1 = idx2 - 1

            if start_dates[idx_line] is pd.NaT:
                start_dates[idx_line] = sub_ts.index[0]

        # Create a DataFrame with the line features
        output_vals = np.vstack([tangents, neighborhoods, start_dates,
                                 flipped, intersections, tangent_rightmost, tangent_leftmost]).T
        columns = ['tangent', 'neighborhood', 'start_date', 
                   'flipped', 'intersections', 'tangent_rightmost', 'tangent_leftmost']
        df = pd.DataFrame(output_vals, columns=columns)
        df_features = pd.DataFrame(output_vals, columns=columns)

        # Combine the features with the original DataFrame
        df = pd.concat([lines.reset_index(drop=True), df_features], axis=1)
        return df

    def calculate_score(self, features, x_obs):
        """ Calculate the score of the trendline - higher numbers are better"""
        
        dT_F = x_obs - features.tangent_rightmost
        dT_L = x_obs - features.tangent_leftmost
        dTT = features.tangent_rightmost - features.tangent_leftmost
        dT_F[np.isnat(dT_F)] = dT_F[np.isnan(dT_F)] = pd.Timedelta(seconds=24 * 3600 * 36500)
        dT_L[np.isnat(dT_L)] = dT_L[np.isnan(dT_L)] = pd.Timedelta(seconds=24 * 3600 * 36500)
        dTT[np.isnat(dT_L)] = dT_L[np.isnan(dT_L)] = pd.Timedelta(seconds=24 * 3600 * 36500)
        
        scores = features.neighborhood.values * \
                 np.power(features.tangent.values, 2) * \
                 (dTT / (dT_L + pd.Timedelta(seconds=120)))
        
        if 'x_R' in features.columns and 'x_L' in features.columns:
            dX = features.x_R - features.x_L
            scores *= (dX / (dT_F + pd.Timedelta(seconds=120)))

        scores[features.start_date.values == x_obs] = 0
        scores[features.tangent.values < 2] = 0
        return scores
        
    def plot_features(self, features, x_obs, direction, n_rows=5, n_cols=3, xlim_R=None):
        """ Method to plot the best scoring features. """
        if direction not in (+1, -1):
            raise ValueError('Argument "direction" must be +1 or -1.')
        
        # Restrict features to the appropriate direction and x_obs
        features = features.loc[features.x_R <= x_obs].query(f'direction == {direction}').copy()
        features.sort_values('score', ascending=False, inplace=True)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        date_index = pd.DatetimeIndex(self.ohlc_obj.index)

        # Get the underlying time series
        if direction == 1:
            underlier_ts = self.ohlc_obj.high
        else:
            underlier_ts = self.ohlc_obj.low

        if xlim_R is not None:
            y_min = underlier_ts[:xlim_R].min()
            y_max = underlier_ts[:xlim_R].max()
        else:
            y_min = underlier_ts.min()
            y_max = underlier_ts.max()
        dY = y_max - y_min

        ctr = 0
        for row in range(n_rows):
            for col in range(n_cols):
                ax = axes[ctr]
                if ctr < features.shape[0]:
                    f = features.iloc[ctr,:]
                    if f.score > 0:
                        L = Line(point=(f.x_0, f.y_0), slope=f.slope, 
                                 timedelta=pd.Timedelta(seconds=1))
                        L.plot(date_index, ax=ax)

                        if 'x_L' in f:
                            ax.axvline(f.x_L, color='red', linestyle='-.')
                            
                        if 'x_R' in f:
                            ax.axvline(f.x_R, color='green', linestyle='-.')

                        if 'start_date' in f:
                            ax.axvline(f.start_date, color='orange', linestyle='-.')

                        ax.axvline(x_obs, color='black', linestyle='-.')

                        underlier_ts.plot(ax=ax, color='blue')
                        ax.set_ylim(y_min - 0.2 * dY, y_max + 0.2 * dY)
                        if xlim_R is not None:
                            ax.set_xlim(self.ohlc_obj.index[0], xlim_R)

                ctr += 1


def calculate_prominence(ohlc_obj, x_obs, y, step_size, atol=None):
    """ Get the biggest prominence of the line
        This is defined to be the maximum of the minimal separation on the left/right
        for any relative maximum or minimum when the prices enter the region.
    """
    if atol is None:
        atol = DEFAULT_ABS_TOL
    
    idx_obs = np.where(ohlc_obj.index == x_obs)[0][0]

    idx_region_below = ohlc_obj.high.values[:idx_obs+1] < y - step_size - atol
    idx_region_above = ohlc_obj.low.values[:idx_obs+1] > y + step_size + atol

    idx_regions = np.zeros((idx_obs+1,), dtype=int)
    idx_regions[idx_region_above] = 1
    idx_regions[idx_region_below] = -1

    idx_region_boundary = np.hstack([idx_regions[1:] != idx_regions[:-1], True])
    timestamps = ohlc_obj.timestamps[:idx_obs+1][idx_region_boundary]
    region_lengths = np.hstack([timestamps[0] - ohlc_obj.timestamps[0] + 1,
                                timestamps[1:] - timestamps[:-1]])
    region_values = idx_regions[idx_region_boundary]
    region_values, timestamps

    # Keep just the values/lengths of the regions far from the line
    outer_region_lengths = region_lengths[region_values != 0]
    outer_region_values = region_values[region_values != 0]

    # Get the biggest prominence of the line
    # This is defined to be the maximum of the minimal separation on the left/right
    #   for any relative maximum or minimum when the prices enter the region
    #   around the horizontal line
    heap = [-outer_region_lengths[0]]
    prominence = 0
    t = 1
    for t in range(outer_region_values.size):
        if outer_region_values[t-1] == outer_region_values[t]:
            heapq.heappush(heap, -outer_region_lengths[t])
        else:
            if len(heap) >= 2:
                _ = heapq.heappop(heap)
                prominence = max(prominence, -heapq.heappop(heap))

            heap = [-outer_region_lengths[t]]

    if len(heap) >= 2:
        _ = heapq.heappop(heap)
        prominence = max(prominence, -heapq.heappop(heap))
    
    return prominence

def get_line_region_classification(ohlc_obj, idx_obs, y, band_width, atol=None):
    """Determine if the prices are in the region near the line. """
    if atol is None:
        atol = DEFAULT_ABS_TOL

    mask_below_inner_region = ohlc_obj.high.values[:idx_obs+1] < y - band_width - atol
    mask_above_inner_region = ohlc_obj.low.values[:idx_obs+1] > y + band_width + atol
    mask_in_inner_region = ~mask_below_inner_region & ~mask_above_inner_region
    
    # Create array of -1, 0, +1 to indicate if prices are below, in, or above the line neighborhood
    all_line_class = np.zeros((idx_obs+1,), dtype=int)
    all_line_class[mask_below_inner_region] = -1
    all_line_class[mask_above_inner_region] = +1
    return all_line_class

def find_right_boundaries(ohlc_obj, x_obs, y, band_width, min_time_for_breach=10, atol=None):
    """ Find the index of the boundaries separating regions where prices are above/below the line.
    
        Returns a tuple (idx_right_boundaries, bounding_direction)
            idx_right_boundaries is an array of indices of the right boundaries between regions
            bounding_direction is +/-1 depending on whether the prices at idx_right_boundaries
                are above/below the line

        Arguments:
            ohlc_obj: (OHLC) object that contains the time series info
            x_obs: (Timestamp) the observation date/time
            y: (float) the level of the horizontal line for which features are to be calculated
            band_width: (float) the price is defined to be in the neighborhood of 'y' if
                the high or low price of the bar is in the region y +/- band_width.
            min_time_for_breach: (float) Number of seconds required before a breach is declared
            atol: (float) the absolute tolerance for floating point operations/comparisons    
    """
    if atol is None:
        atol = DEFAULT_ABS_TOL

    # Get the index of the observation point
    idx_obs = np.where(ohlc_obj.index == x_obs)[0][0]
    index = np.arange(0, idx_obs + 1)

    # Determine if the prices are in the region near the line
    all_line_class = get_line_region_classification(ohlc_obj, 
            idx_obs=idx_obs, y=y, band_width=band_width, atol=atol)
    
    idx_outside_inner_region = index[all_line_class != 0]
    idx_first = idx_outside_inner_region.min()

    # Find the right-hand index of every region where the line class changes
    all_idx_rb = index[np.hstack([all_line_class[1:] != all_line_class[:-1], True])]
    all_separation = np.hstack([all_idx_rb[0], all_idx_rb[1:] - all_idx_rb[:-1]])

    # Find the right-hand index of regions that are bigger than the min. allowable size
    idx_rb = all_idx_rb[all_separation >= min_time_for_breach]

    # Only keep boundaries where the class changes
    idx_right_boundaries_rev = []
    for k, idx_curr in enumerate(idx_rb[::-1]):
        dir_curr = all_line_class[idx_curr]
        if k == 0 or dir_last != dir_curr:
            idx_right_boundaries_rev.append(idx_curr)
            idx_last = idx_curr
            dir_last = dir_curr

    idx_right_boundaries = np.array(idx_right_boundaries_rev[::-1], dtype=int)
    bounding_direction = all_line_class[idx_right_boundaries]
    return idx_right_boundaries, bounding_direction

def calculate_features_for_horizontal_line(ohlc_obj, x_obs, y, band_width,
                                           max_intersections=4, 
                                           min_time_for_breach=10,
                                           atol=None):
    """ Find all lines of support/resistance at a given observation point.
    
        The algorithm goes through several steps:
            1. For all points, determine whether they are above or below the line,
                or whether they are in the neighborhood of the line.
            2. Find the right-hand boundary of all regions that are either above or
                below the line, but not in the region of the line. Ignore any boundaries
                if they demarcate a region that is shorter than 'min_time_for_breach.'
            3. Find whether the prices are above or below the line in each region 
                obtained in (2)
            4. From (3), determine whether the prices have been deflected off of the
                line or have intersected it at each of the boundaries. Note the 
                points of intersection.
            5. Going backward from the observation time x_obs, loop through each of the
                intersection points. Within each sub-region where no intersection
                has occurred, calculate statistics like the number of deflections by
                the line and the number of points in the neighborhood of the line
            6. Finally, calculate the prominence of the line, which takes into account
                all points.
                
        Returns a dict of calculated features.

        Arguments:
            ohlc_obj: (OHLC) object that contains the time series info
            x_obs: (Timestamp) the observation date/time
            y: (float) the level of the horizontal line for which features are to be calculated
            band_width: (float) the price is defined to be in the neighborhood of 'y' if
                the high or low price of the bar is in the region y +/- band_width.
            max_intersections: (int) The maximum number of intersections for which it will
                calculate features, going backward starting from x_obs. After more than
                this number of intersections have occurred, the calculation ends.
            min_time_for_breach: (float) Number of seconds required before a breach is declared
            atol: (float) the absolute tolerance for floating point operations/comparisons
    """
    if atol is None:
        atol = DEFAULT_ABS_TOL

    idx_right_boundaries, bounding_direction = find_right_boundaries(ohlc_obj, x_obs=x_obs, y=y, 
                     band_width=band_width, min_time_for_breach=min_time_for_breach, atol=atol)

    idx_obs = np.where(ohlc_obj.index == x_obs)[0][0]
    low_vals = ohlc_obj.low.values[:idx_obs+1]    
    high_vals = ohlc_obj.high.values[:idx_obs+1]
    index = np.arange(idx_obs+1)
    
    # Only keep boundaries between regions with prices above or below line
    idx_right_boundaries = idx_right_boundaries[bounding_direction != 0]
    bounding_direction = bounding_direction[bounding_direction != 0]

    # Find the index at the right side of each region, and whether 
    #   a deflection or intersection occurred
    idx_bdry = bounding_direction[1:] != bounding_direction[:-1]
    idx_deflections = idx_right_boundaries[:-1][~idx_bdry]
    idx_intersections = idx_right_boundaries[:-1][idx_bdry]

    # Get the direction in each of the major regions
    regional_directions = bounding_direction[:-1][idx_bdry]
    finished = False
    p = 1
    while not finished:
        if bounding_direction[-p] != 0:
            regional_directions = np.hstack([regional_directions,
                                             bounding_direction[-p]])
            finished = True
        p += 1

    # Add the first index to the list of regional separators, because the number of
    # distinct regions should be 1 more than the number of intersections
    idx_left_major_boundaries = np.hstack([0, 1 + idx_intersections])

    # Determine if the prices are in the region near the line
    all_line_class = get_line_region_classification(ohlc_obj, 
            idx_obs=idx_obs, y=y, band_width=band_width, atol=atol)

    # Go through the intersection points in reverse order, starting at x_obs
    j = 0
    n_deflections = -1 * np.ones((max_intersections + 1,), dtype=int)
    directions = -1 * np.ones((max_intersections + 1,), dtype=int)
    n_neighbors = -1 * np.ones((max_intersections + 1,), dtype=int)
    n_violations = -1 * np.ones((max_intersections + 1,), dtype=int)
    violations_sum_1 = np.nan * np.empty((max_intersections + 1,), dtype=float)
    violations_sum_2 = np.nan * np.empty((max_intersections + 1,), dtype=float)
    x_right = pd.NaT * np.empty((max_intersections + 1,))
    x_left = pd.NaT * np.empty((max_intersections + 1,))
    durations = -1 * np.ones((max_intersections + 1,), dtype=int)
    idx_right = idx_obs
    for j, idx_left in enumerate(idx_left_major_boundaries[::-1]):
        if j >= max_intersections:
            break

        # Find the number of deflections/intersections between the left/right boundaries
        mask_bdry = (idx_left <= idx_right_boundaries) & (idx_right_boundaries < idx_right)
        n_deflections[j] = mask_bdry.sum()
        directions[j] = regional_directions[-j-1]
        durations[j] = idx_right - idx_left + 1
        x_left[j] = ohlc_obj.index[idx_left]
        x_right[j] = ohlc_obj.index[idx_right]

        # Find the number of points in the neighborhood of the line btwn left/right boundaries
        mask_full = (idx_left < idx_right_boundaries) & (idx_right_boundaries < idx_right)
        idx_region = (idx_left < index) & (index < idx_right)
        n_neighbors[j] = (0 == all_line_class[idx_region]).sum()

        if directions[j] == 1:
            idx_v = low_vals[idx_region] <= y - atol
            vals = np.abs(low_vals[idx_region][idx_v] - y)
        elif directions[j] == -1:
            idx_v = high_vals[idx_region] >= y + atol
            vals = np.abs(high_vals[idx_region][idx_v] - y)
        else:
            raise ValueError('Direction should be +1 or -1')

        n_violations[j] = np.sum(idx_v)
        violations_sum_1[j] = vals.sum() / band_width
        violations_sum_2[j] = np.power(vals, 2).sum() / np.power(band_width, 2)
            
        idx_right = idx_left - 1
        j += 1

    # Find the maximum prominence of the line
    prominence = calculate_prominence(ohlc_obj, x_obs=x_obs, y=y,
                                      step_size=band_width)

    # Create a dict for outputting the results
    results = dict(slope=0.0, 
                   x_0=ohlc_obj.index[0], y_0=y,
                   x_R=x_obs, y_R=y, 
                   prominence=prominence,
                   directions=directions,
                   n_neighbors=n_neighbors,
                   n_deflections=n_deflections,
                   n_violations=n_violations,
                   violations_sum_1=violations_sum_1,
                   violations_sum_2=violations_sum_2,
                   durations=durations,
                   x_left=x_left,
                   x_right=x_right,                   
                  )

    return results

def find_all_support_and_resistence_lines(ohlc_obj, x_obs, step_size, band_width,
                                          min_time_for_breach=10,
                                          max_intersections=4, atol=None):
    """ Find all lines of support/resistance at a given observation point x_obs.
    
        Returns a pandas DataFrame of results
        
        Arguments:
            ohlc_obj: (OHLC) object that contains the time series info
            x_obs: (Timestamp) the observation date/time
            step_size: (float) how large of steps to take when looping through the
                set of horizontal lines.
            band_width: (float) the price is defined to be in the neighborhood of 'y' if
                the high or low price of the bar is in the region y +/- band_width.
            max_intersections: (int) The maximum number of intersections for which it will
                calculate features, going backward starting from x_obs. After more than
                this number of intersections have occurred, the calculation ends.
            min_time_for_breach: (float) Number of seconds required before a breach is declared
            atol: (float) the absolute tolerance for floating point operations/comparisons
    """
    # Get the index of the observation point
    idx_obs = np.where(ohlc_obj.index == x_obs)[0][0]

    # Get the max and min y-values during the observation period
    y_min = ohlc_obj.low.values[:idx_obs+1].min()
    y_max = ohlc_obj.high.values[:idx_obs+1].max()

    # Loop through all y-values to test if they are a support/resistance line
    hlines = []
    for y in np.arange(y_min, y_max + step_size, step_size):        
        results = calculate_features_for_horizontal_line(ohlc_obj, x_obs=x_obs, y=y,
                                                         band_width=band_width,
                                                         max_intersections=max_intersections, 
                                                         min_time_for_breach=min_time_for_breach,
                                                         atol=atol)

        hlines.append(results)

    # Combine the results into a DataFrame
    df_hlines = pd.DataFrame.from_dict(hlines)
    return df_hlines

def calculate_line_feature_score(ohlc_obj, df_features, x_obs, max_intersections):
    """ Calculate a score for a support/resistance line based on features.
    """
    T = (x_obs - ohlc_obj.index[0]) / pd.Timedelta(seconds=1)

    scores = []
    x_scores = []
    all_scores = []
    for j in range(df_features.shape[0]):
        row = df_features.iloc[j]
        max_score = -1
        max_score_x = ohlc_obj.index[0]
        score_info = dict()
        for k in range(1, max_intersections + 1):
            if row.n_neighbors[k-1] > -1 and row.directions[k-1] != 0:
                n_deflections = row.n_deflections[:k]            
                n_neighbors = row.n_neighbors[:k]
                n_violations = row.n_violations[:k]
                violations_sum_1 = row.violations_sum_1[:k]
                violations_sum_2 = row.violations_sum_2[:k]            
                durations = row.durations[:k]

                score_duration = durations/T
                score_deflection = 1e4 * (1 + n_deflections) / T
                score_violations = 100 / np.power(1 + violations_sum_2 / (1 + n_neighbors), 4)
                score_neighbors = 100 * (1 + n_neighbors) / T
                total = score_duration * score_deflection * score_violations * score_neighbors
                score = np.max(total)
                if score > max_score:
                    max_score = score
                    idx_max = np.argmax(total)
                    max_score_x = row.x_left[idx_max]
                    score_info = dict(duration=score_duration[idx_max], deflection=score_deflection[idx_max],
                                      violations=score_violations[idx_max], neighbors=score_neighbors[idx_max])

        scores.append(max_score)
        x_scores.append(max_score_x)
        all_scores.append(score_info)

    df = df_features.copy()
    df['score'] = scores
    df['score_x'] = x_scores
    df['score_info'] = all_scores
    return df
