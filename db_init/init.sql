

drop table if exists zipcode_demo_stage;

create table zipcode_demo_stage(
    ppltn_qty  float
    ,urbn_ppltn_qty float
    ,sbrbn_ppltn_qty float
    ,farm_ppltn_qty float
    ,non_farm_qty float
    ,medn_hshld_incm_amt float
    ,medn_incm_per_prsn_amt float
    ,hous_val_amt float
    ,edctn_less_than_9_qty float
    ,edctn_9_12_qty float
    ,edctn_high_schl_qty float
    ,edctn_some_clg_qty float
    ,edctn_assoc_dgre_qty float
    ,edctn_bchlr_dgre_qty float
    ,edctn_prfsnl_qty float
    ,per_urbn float
    ,per_sbrbn float
    ,per_farm float
    ,per_non_farm float
    ,per_less_than_9 float
    ,per_9_to_12 float
    ,per_hsd float
    ,per_some_clg float
    ,per_assoc float
    ,per_bchlr float
    ,per_prfsnl float
    ,zipcode integer PRIMARY KEY
);


COPY zipcode_demo_stage FROM '/datafiles/zipcode_demographics.csv' DELIMITER ',' CSV HEADER;


drop table if exists zipcode_demo;
create table zipcode_demo(
    ppltn_qty  integer
    ,urbn_ppltn_qty integer
    ,sbrbn_ppltn_qty integer
    ,farm_ppltn_qty integer
    ,non_farm_qty integer
    ,medn_hshld_incm_amt float
    ,medn_incm_per_prsn_amt float
    ,hous_val_amt float
    ,edctn_less_than_9_qty float
    ,edctn_9_12_qty float
    ,edctn_high_schl_qty float
    ,edctn_some_clg_qty float
    ,edctn_assoc_dgre_qty float
    ,edctn_bchlr_dgre_qty float
    ,edctn_prfsnl_qty float
    ,per_urbn float
    ,per_sbrbn float
    ,per_farm float
    ,per_non_farm float
    ,per_less_than_9 float
    ,per_9_to_12 float
    ,per_hsd float
    ,per_some_clg float
    ,per_assoc float
    ,per_bchlr float
    ,per_prfsnl float
    ,zipcode integer PRIMARY KEY
);

insert into zipcode_demo
select 
    cast(ppltn_qty as  integer) ppltn_qty
    ,cast(urbn_ppltn_qty as integer) urbn_ppltn_qty
    ,cast(sbrbn_ppltn_qty as integer) sbrbn_ppltn_qty
    ,cast(farm_ppltn_qty as integer) farm_ppltn_qty
    ,cast(non_farm_qty as integer) non_farm_qty
    ,medn_hshld_incm_amt 
    ,medn_incm_per_prsn_amt 
    ,hous_val_amt 
    ,edctn_less_than_9_qty 
    ,edctn_9_12_qty 
    ,edctn_high_schl_qty 
    ,edctn_some_clg_qty 
    ,edctn_assoc_dgre_qty 
    ,edctn_bchlr_dgre_qty 
    ,edctn_prfsnl_qty 
    ,per_urbn 
    ,per_sbrbn 
    ,per_farm 
    ,per_non_farm 
    ,per_less_than_9 
    ,per_9_to_12 
    ,per_hsd 
    ,per_some_clg 
    ,per_assoc 
    ,per_bchlr 
    ,per_prfsnl 
    ,zipcode
 from zipcode_demo_stage;
