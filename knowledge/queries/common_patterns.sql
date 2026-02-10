-- ==================== QUERY PATTERNS FOR TELECOM ANALYTICS ====================

-- PATTERN 1: Get latest subscriber data (most recent insertion_date)
-- USE CASE: Base query for any subscriber analysis
SELECT * 
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY subs_id ORDER BY Insertion_Date DESC) AS rn
    FROM analytic_models.Subscriber_Profile
    WHERE Subscriber_Status IS NOT NULL
) t 
WHERE rn = 1 AND GOV = 'Giza'
-- Result: Most recent, deduplicated subscriber records

-- PATTERN 2: ARPU by Technology Segment
-- USE CASE: "Which technology has highest ARPU?"
SELECT 
    Current_Technology,
    COUNT(DISTINCT subs_id) as subscriber_count,
    AVG(ARPU) as avg_arpu,
    SUM(ARPU) as total_revenue,
    MIN(ARPU) as min_arpu,
    MAX(ARPU) as max_arpu,
    STDDEV(ARPU) as arpu_variance
FROM (
    SELECT * FROM analytic_models.Subscriber_Profile
    WHERE GOV = 'Giza' AND Subscriber_Status IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY subs_id ORDER BY Insertion_Date DESC) = 1
) t
GROUP BY 1
ORDER BY avg_arpu DESC;
-- Result: Technology profitability ranking

-- PATTERN 3: Churn Risk Segmentation
-- USE CASE: "What's our at-risk subscriber base?"
SELECT 
    Stability_Name,
    COUNT(*) as count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct_of_total,
    AVG(ARPU) as avg_arpu,
    AVG(Tenure_Days) as avg_tenure
FROM (
    SELECT * FROM analytic_models.Subscriber_Profile
    WHERE GOV = 'Giza' AND Subscriber_Status IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY subs_id ORDER BY Insertion_Date DESC) = 1
) t
GROUP BY 1
ORDER BY count DESC;
-- Result: Stability distribution for targeting

-- PATTERN 4: Geography Performance Comparison
-- USE CASE: "Which governorate has best ARPU?"
SELECT 
    GOV,
    COUNT(*) as subscribers,
    AVG(ARPU) as avg_arpu,
    AVG(Tenure_Days) as avg_tenure,
    COUNT(CASE WHEN Stability_Name = 'Stable' THEN 1 END) as stable_count,
    COUNT(CASE WHEN Stability_Name = 'At-Risk' THEN 1 END) as at_risk_count
FROM (
    SELECT * FROM analytic_models.Subscriber_Profile
    WHERE Subscriber_Status IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY subs_id ORDER BY Insertion_Date DESC) = 1
) t
GROUP BY 1
ORDER BY avg_arpu DESC;
-- Result: Geographic performance metrics

-- PATTERN 5: Revenue Concentration (Pareto Analysis)
-- USE CASE: "Which segment drives most revenue?"
SELECT 
    Current_Technology,
    SUM(ARPU) as revenue,
    SUM(ARPU) * 100.0 / SUM(SUM(ARPU)) OVER () as pct_of_total_revenue,
    SUM(SUM(ARPU)) OVER (ORDER BY SUM(ARPU) DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * 100.0 / 
    SUM(SUM(ARPU)) OVER () as cumulative_pct
FROM (
    SELECT * FROM analytic_models.Subscriber_Profile
    WHERE GOV = 'Giza' AND Subscriber_Status IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY subs_id ORDER BY Insertion_Date DESC) = 1
) t
GROUP BY 1
ORDER BY revenue DESC;
-- Result: Identify top revenue drivers

-- PATTERN 6: FTTH Migration Opportunity
-- USE CASE: "How many high-value non-FTTH customers can we upgrade?"
SELECT 
    COUNT(*) as migration_candidates,
    AVG(ARPU) as current_avg_arpu,
    AVG(ARPU) * 1.4 as potential_arpu_after_ftth,
    SUM(ARPU) as current_revenue,
    SUM(ARPU) * 1.4 as potential_revenue,
    AVG(Tenure_Days) as avg_tenure_of_candidates
FROM (
    SELECT * FROM analytic_models.Subscriber_Profile
    WHERE GOV = 'Giza' 
      AND Subscriber_Status IS NOT NULL
      AND Current_Technology != 'FTTH'
      AND ARPU > 100
      AND Stability_Name = 'Stable'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY subs_id ORDER BY Insertion_Date DESC) = 1
) t;
-- Result: Revenue expansion opportunities

-- PATTERN 7: Demographic Segmentation
-- USE CASE: "Which age/gender segment is most valuable?"
SELECT 
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 50 THEN '35-49'
        WHEN age < 65 THEN '50-64'
        ELSE '65+'
    END as age_group,
    Gender,
    COUNT(*) as subscribers,
    AVG(ARPU) as avg_arpu,
    MAX(ARPU) as max_arpu,
    AVG(Tenure_Days) as avg_tenure
FROM (
    SELECT * FROM analytic_models.Subscriber_Profile
    WHERE GOV = 'Giza' AND Subscriber_Status IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY subs_id ORDER BY Insertion_Date DESC) = 1
) t
GROUP BY 1, 2
ORDER BY avg_arpu DESC;
-- Result: Premium segment identification

-- PATTERN 8: Population Density Analysis
-- USE CASE: "Which area (Urban/Suburban/Rural) has growth potential?"
SELECT 
    PopName,
    COUNT(*) as subscriber_count,
    AVG(ARPU) as avg_arpu,
    COUNT(CASE WHEN Stability_Name = 'Stable' THEN 1 END) * 100.0 / COUNT(*) as stability_rate,
    SUM(ARPU) as total_revenue
FROM (
    SELECT * FROM analytic_models.Subscriber_Profile
    WHERE GOV = 'Giza' AND Subscriber_Status IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY subs_id ORDER BY Insertion_Date DESC) = 1
) t
GROUP BY 1
ORDER BY avg_arpu DESC;
-- Result: Area-level performance and growth potential
