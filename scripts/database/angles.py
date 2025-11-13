import gc

import mysql.connector

DB_Config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'asd515359',
    'database': 'config_db',
    'port' : 3306
}

R_start,R_end,R_step = 5500,500,-500
theta_start,theta_end,theta_step = 0,360,30
phi_start,phi_end,phi_step = 30,90,5
def main():
    conn = mysql.connector.connect(**DB_Config)
    cursor = conn.cursor()
    cursor.execute("select max(id) from angles")
    max_id = cursor.fetchone()[0]
    if max_id is None:
        max_id = 0
    start_id = max_id + 1
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS angles(
                   R INT NOT NULL,
                   theta INT NOT NULL,
                   phi INT NOT NULL,
                   id INT NOT NULL)ENGINE=InnoDB;
                   """)

    sql_insert = "INSERT INTO angles (R,theta,phi,id) VALUES (%s,%s,%s,%s)"

    batch_size = 100
    count = 0
    dataInsert = []
    for R in range(R_start, R_end, R_step):
        for phi in range(phi_start, phi_end, phi_step):
            for theta in range(theta_start,theta_end,theta_step):
                dataInsert.append((R,theta,phi,start_id))
                start_id += 1
                if count % batch_size == 0:
                    cursor.executemany(sql_insert,dataInsert)
                    conn.commit()
                    dataInsert.clear()
                    count = 0
    if dataInsert:
        cursor.executemany(sql_insert, dataInsert)
        conn.commit()
        dataInsert.clear()
        # batch_size = 100
        # count = 0
        # values = []
        #
        # for theta in range(theta_start,theta_end,theta_step):
        #     for phi in range(phi_start,phi_end,phi_step):
        #         values.append(theta,phi)
        #         count += 1
        #         if count % batch_size == 0:
        #             cursor.executemany(sql_insert,values)
        #             conn.commit()
        #             values.clear()

    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()