FROM node:20-alpine

# Set the working directory
WORKDIR /app

# Copy package.json to leverage Docker layer caching
COPY package.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application files
COPY . .

# Expose port 8502 (consistent with original streamlit port)
EXPOSE 8502

# Run the Vite development server mapped to 8502
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "8502"]
