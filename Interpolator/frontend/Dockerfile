# frontend/Dockerfile

# install dependencies

# if app not exist, use current dir
FROM node:20-alpine AS deps
WORKDIR /app  


# based on package.json install all dependencies
COPY package.json package-lock.json ./
RUN npm install

# build the Next.js app
FROM node:20-alpine AS builder
WORKDIR /app

COPY . .
COPY --from=deps /app/node_modules ./node_modules

ARG NEXT_PUBLIC_BACKEND_URL
ENV NEXT_PUBLIC_BACKEND_URL=$NEXT_PUBLIC_BACKEND_URL

RUN npm run build

# run the production server
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/package.json ./package.json
COPY --from=deps /app/node_modules ./node_modules

EXPOSE 3000
CMD ["npm", "start"]
