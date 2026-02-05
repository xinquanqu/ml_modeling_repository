FROM postgres:15-alpine

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/

# Install useful extensions
RUN apk add --no-cache postgresql15-contrib

EXPOSE 5432
