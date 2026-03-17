-- Create the database
CREATE DATABASE IF NOT EXISTS ecommerce_bot_test;
USE ecommerce_bot_test;

-- Drop existing tables if any
SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
SET FOREIGN_KEY_CHECKS = 1;

-- Customers table
CREATE TABLE customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INT DEFAULT 0
);

-- Orders table
CREATE TABLE orders (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status ENUM('pending', 'shipped', 'delivered', 'cancelled') DEFAULT 'pending',
    total_amount DECIMAL(10, 2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Order Items table
CREATE TABLE order_items (
    order_item_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Sample Data
INSERT INTO customers (first_name, last_name, email) VALUES
('John', 'Doe', 'john.doe@example.com'),
('Jane', 'Smith', 'jane.smith@example.com');

INSERT INTO products (name, description, price, stock_quantity) VALUES
('Laptop', 'High performance laptop', 1200.00, 10),
('Smartphone', 'Latest model smartphone', 800.00, 20),
('Headphones', 'Noise cancelling headphones', 150.00, 30);

-- Stored Procedures
DELIMITER //

CREATE PROCEDURE GetCustomerTotalSpend(IN cust_id INT, OUT total DECIMAL(10, 2))
BEGIN
    SELECT SUM(total_amount) INTO total
    FROM orders
    WHERE customer_id = cust_id;
END //

CREATE PROCEDURE ProcessOrder(IN cust_id INT, IN prod_id INT, IN qty INT)
BEGIN
    DECLARE unit_p DECIMAL(10, 2);
    DECLARE last_id INT;
    
    -- Get product price
    SELECT price INTO unit_p FROM products WHERE product_id = prod_id;
    
    -- Create order
    INSERT INTO orders (customer_id, total_amount, status) 
    VALUES (cust_id, unit_p * qty, 'pending');
    
    SET last_id = LAST_INSERT_ID();
    
    -- Create order item
    INSERT INTO order_items (order_id, product_id, quantity, unit_price)
    VALUES (last_id, prod_id, qty, unit_p);
    
    -- Reduce stock
    UPDATE products SET stock_quantity = stock_quantity - qty WHERE product_id = prod_id;
END //

DELIMITER ;
