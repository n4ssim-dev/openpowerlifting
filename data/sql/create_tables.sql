-- Star schema fitted to openpowerlifting.csv (facts) and meets.csv (dim_rencontre) from the Kaggle dataset dansbecker/powerlifting-database

CREATE TABLE dim_rencontre(
   id_rencontre INT,               -- MeetID
   meet_path VARCHAR(60),          -- MeetPath
   federation VARCHAR(30),         -- Federation
   date_rencontre DATE,            -- Date (YYYY-MM-DD)
   pays VARCHAR(30),                -- MeetCountry
   etat VARCHAR(10),                -- MeetState (nullable, ~35% empty)
   ville VARCHAR(60),               -- MeetTown
   nom_rencontre VARCHAR(200),      -- MeetName
   PRIMARY KEY(id_rencontre)
);

CREATE TABLE dim_powerlifter(
   id_powerlifter VARCHAR(100),    -- Name (only identifier available in the source data)
   sexe VARCHAR(1),                -- Sex: 'M' or 'F'
   PRIMARY KEY(id_powerlifter)
);

CREATE TABLE dim_division(
   id_division VARCHAR(50),        -- Division label (e.g. "Junior Men 14-15", "M-O-RC"...)
   PRIMARY KEY(id_division)
);

CREATE TABLE dim_equipement(
   id_equipement VARCHAR(20),      -- Equipment: Raw, Wraps, Single-ply, Multi-ply
   PRIMARY KEY(id_equipement)
);

CREATE TABLE fait_performance(
   id_performance INT,                     -- surrogate key generated at load time (no natural key: a lifter can enter several times in one meet)
   age DECIMAL(4,1),                       -- Age: varies per meet, half-year values exist (e.g. 10.5)
   poids_kg DECIMAL(15,2),                 -- BodyweightKg: varies per meet, belongs to the fact not the lifter
   categorie_poids_kg VARCHAR(10),         -- WeightClassKg: not numeric-only, e.g. "110+", "82.5"
   coeff_wilks DECIMAL(15,2),
   total_kg DECIMAL(15,2),
   place VARCHAR(3),                       -- numeric rank OR 'DQ','DD','NS','G'
   squat_4kg DECIMAL(15,2),
   bench_4kg DECIMAL(15,2),
   deadlift_4kg DECIMAL(15,2),
   best_squat_kg DECIMAL(15,2),
   best_bench_kg DECIMAL(15,2),
   best_deadlift_kg DECIMAL(15,2),
   id_rencontre INT NOT NULL,
   id_division VARCHAR(50),                -- nullable: ~4% of rows have no Division in the source data
   id_powerlifter VARCHAR(100) NOT NULL,
   id_equipement VARCHAR(20) NOT NULL,
   PRIMARY KEY(id_performance),
   FOREIGN KEY(id_rencontre) REFERENCES dim_rencontre(id_rencontre),
   FOREIGN KEY(id_division) REFERENCES dim_division(id_division),
   FOREIGN KEY(id_powerlifter) REFERENCES dim_powerlifter(id_powerlifter),
   FOREIGN KEY(id_equipement) REFERENCES dim_equipement(id_equipement)
);

-- Distinct from fait_performance's grain: a lifter can appear several times in the
-- same meet (different divisions/weight classes), so this bridge deduplicates
-- "which lifters took part in which meet" as a pure M:N relationship.
CREATE TABLE participation_rencontre(
   id_rencontre INT,
   id_powerlifter VARCHAR(100),
   PRIMARY KEY(id_rencontre, id_powerlifter),
   FOREIGN KEY(id_rencontre) REFERENCES dim_rencontre(id_rencontre),
   FOREIGN KEY(id_powerlifter) REFERENCES dim_powerlifter(id_powerlifter)
);
