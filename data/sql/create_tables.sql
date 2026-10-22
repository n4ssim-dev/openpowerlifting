CREATE TABLE dim_rencontre(
   id_rencontre INT,               
   meet_path VARCHAR(60),         
   federation VARCHAR(30),        
   date_rencontre DATE,           
   pays VARCHAR(30),                
   etat VARCHAR(10),                
   ville VARCHAR(60),               
   nom_rencontre VARCHAR(200),      
   PRIMARY KEY(id_rencontre)
);

CREATE TABLE dim_powerlifter(
   id_powerlifter VARCHAR(100),
   sexe VARCHAR(1),
   PRIMARY KEY(id_powerlifter)
);

CREATE TABLE dim_division(
   id_division VARCHAR(50),
   PRIMARY KEY(id_division)
);

CREATE TABLE dim_equipement(
   id_equipement VARCHAR(20),
   PRIMARY KEY(id_equipement)
);

CREATE TABLE fait_performance(
   id_performance INT,                     
   age DECIMAL(4,1),                     
   poids_kg DECIMAL(15,2),                
   categorie_poids_kg VARCHAR(10),         
   coeff_wilks DECIMAL(15,2),
   total_kg DECIMAL(15,2),
   place VARCHAR(3),                      
   squat_4kg DECIMAL(15,2),
   bench_4kg DECIMAL(15,2),
   deadlift_4kg DECIMAL(15,2),
   best_squat_kg DECIMAL(15,2),
   best_bench_kg DECIMAL(15,2),
   best_deadlift_kg DECIMAL(15,2),
   id_rencontre INT NOT NULL,
   id_division VARCHAR(50),                
   id_powerlifter VARCHAR(100) NOT NULL,
   id_equipement VARCHAR(20) NOT NULL,
   PRIMARY KEY(id_performance),
   FOREIGN KEY(id_rencontre) REFERENCES dim_rencontre(id_rencontre),
   FOREIGN KEY(id_division) REFERENCES dim_division(id_division),
   FOREIGN KEY(id_powerlifter) REFERENCES dim_powerlifter(id_powerlifter),
   FOREIGN KEY(id_equipement) REFERENCES dim_equipement(id_equipement)
);

CREATE TABLE participation_rencontre(
   id_rencontre INT,
   id_powerlifter VARCHAR(100),
   PRIMARY KEY(id_rencontre, id_powerlifter),
   FOREIGN KEY(id_rencontre) REFERENCES dim_rencontre(id_rencontre),
   FOREIGN KEY(id_powerlifter) REFERENCES dim_powerlifter(id_powerlifter)
);
