DROP TABLE IF EXISTS evalution;
CREATE TABLE evalution(
    image_id BIGINT NOT NULL,
    f1 double NOT NULL,
    model INT DEFAULT 0
)ENGINE=InnoDB;