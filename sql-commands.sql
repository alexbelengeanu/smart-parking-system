CREATE TABLE AllowedVehicles (
    PlateID int NOT NULL AUTO_INCREMENT,
    PlateNumber varchar(8),
    OwnerName varchar(255),
    PRIMARY KEY (PlateID)
);


INSERT INTO AllowedVehicles (PlateNumber,OwnerName)
VALUES ('TM20EUZ','Alexandru-Florin BELENGEANU');
INSERT INTO AllowedVehicles (PlateNumber,OwnerName)
VALUES ('B121PHD','Marian Dragulescu');
INSERT INTO AllowedVehicles (PlateNumber,OwnerName)
VALUES ('B961TAG','Ana Popovici');